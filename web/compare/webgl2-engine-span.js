// 자체 WebGL2 추론 엔진 (위상 분할 SPAN 레이아웃) — SegGPU 고정 그래프 전용.
//
// 타일판(webgl2-engine-tiled.js)에서 레이아웃만 바꾼 것. 한 프래그먼트가 가로 SPAN개
// 픽셀을 담당해 가중치 페치를 SPAN분의 1로 나눈다 (실측 SPAN4 = 1.9배, span-bench.html).
// 구글이 논문에서 MRT를 강조한 본래 용도가 이것 — 출력 채널이 아니라 출력 픽셀 묶기.
//
//   텐서 [1,C,H,W] → 텍스처 SPAN개, 각 (W/SPAN × ⌈C/4⌉, H)
//   픽셀 x  →  위상 p = x % SPAN, 열 xq = x / SPAN   (x = xq*SPAN + p)
//   모든 셰이더가 SPAN개 어태치먼트(MRT)로 위상 전부를 1패스에 쓴다.
//
// 위상 인덱싱이 컴파일 타임 상수로 떨어지는 게 핵심:
//   출력 x*SPAN+t, 탭 kj → 입력 (x*SPAN+t)*sw + kj - pl = x*SPAN*sw + c,  c = t*sw+kj-pl
//   SPAN*sw가 SPAN의 배수이므로 위상 = c mod SPAN (x와 무관), 열 = x*sw + ⌊c/SPAN⌋
// stride·AvgPool(k=s)도 같은 이유로 정적. Resize만 보간 좌표가 비정수라 런타임 분기.
//
// 타일판과의 차이:
//   - Concat 융합 없음 (소스 3개 × 위상 4 = 샘플러 12개 + 가중치·bias — 위험해서 1차는 복사)
//   - binary 융합 없음 (타일판에서도 기본 off)
//   - fp16 활성 없음 (타일판 실측: 이득 0, 정확도 파괴)
//
// 주의: debug 모드는 draw마다 gl.getError()로 GPU를 동기화시켜 측정을 20배 이상
// 왜곡한다. 정확성 확인에만 켜고 타이밍에는 반드시 끈다.

const SPAN = 4;

const VERT = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const ACT = {
  none: 'v',
  relu: 'max(v, vec4(0.0))',
  sigmoid: '(1.0 / (1.0 + exp(-v)))',
  tanh: 'tanh(v)',
  hardswish: '(v * clamp(v + 3.0, 0.0, 6.0) / 6.0)',
  hardsigmoid: 'clamp(v / 6.0 + 0.5, 0.0, 1.0)',
  clamp01: 'clamp(v, 0.0, 1.0)',
};

// 출력 좌표 → (og, x, y). x는 위상 내 열(xq). OWQ = 출력 W / SPAN.
const HEAD = (OWQ) => `
  int fx = int(gl_FragCoord.x);
  int og = fx / ${OWQ} - uOutOff;
  int x = fx - (og + uOutOff) * ${OWQ};
  int y = int(gl_FragCoord.y);`;
const OFFS = 'uniform int uInOff; uniform int uOutOff;';

const INS = (pfx = 'uIn') =>
  Array.from({ length: SPAN }, (_, t) => `uniform sampler2D ${pfx}${t};`).join(' ');
const OUTS = () =>
  Array.from({ length: SPAN }, (_, t) => `layout(location = ${t}) out vec4 o${t};`).join('\n');

// (t, c) → 위상·열오프셋. c = t*sw + kj - pl 형태의 컴파일 타임 상수.
const phaseOf = (c) => ((c % SPAN) + SPAN) % SPAN;
const dxOf = (c) => Math.floor(c / SPAN);

// splitK: 입력 채널이 많은 conv는 igFrom..igTo만 처리하는 부분 conv를 만들어
// 가산 블렌딩으로 누적한다 (타일판 동일).
// srcs: [{L}] — Concat을 융합했을 때의 입력 소스들. 소스 si의 위상 ph는 uIn{si}_{ph},
// 레이어 오프셋은 uInOff{si}. ig 루프를 소스 경계에서 쪼개 런타임 루프를 유지한다.
// kSplit {CS, chunks, oH}: 프래그먼트 기아 conv용 — 스크래치를 세로로 chunks배 쌓아
// 입력 채널 청크별 부분합을 "병렬로" 계산한다 (blend 누적은 순차 패스라 기아를 못 푼다).
// 청크 인덱스 = y / oH (런타임), 행 = y % oH. 뒤에 REDUCE_SRC가 세로 합산.
function convSrc({ dw, k, s, pad, act, iWQ, iH, iL, oWQ, srcs = null, dil = [1, 1],
                   igFrom = 0, igTo = 0, noTail = false, post = null, kSplit = null }) {
  const [kh, kw] = k, [sh, sw] = s, [pt, pl] = [pad[0], pad[1]];
  const [dh, dv] = dil;                 // 탭 간격 (dilated conv — 위상 수학은 그대로 상수)
  const taps = kh * kw;
  const S = srcs || [{ L: iL }];
  const totL = S.reduce((a, c) => a + c.L, 0);
  const hi = igTo || totL;
  const smp = (si, ph) => (srcs ? `uIn${si}_${ph}` : `uIn${ph}`);
  const offU = (si) => (srcs ? `uInOff${si}` : 'uInOff');

  // 행(ki) 단위로 처리: 같은 행의 kj 탭들이 겹치는 열을 공유하므로 입력 페치를
  // 행당 한 번씩만 한다 (3×3이면 탭별 12페치 → 6페치).
  let body = '';
  for (let ki = 0; ki < kh; ki++) {
    // 이 행에서 (모든 kj × 위상)이 필요로 하는 입력 열 집합
    const uniq = new Map();               // "ph|dx" → 변수명
    const at = (t, kj) => {
      const c = t * sw + kj * dv - pl;
      return `${phaseOf(c)}|${dxOf(c)}`;
    };
    for (let kj = 0; kj < kw; kj++) {
      for (let t = 0; t < SPAN; t++) {
        const kk = at(t, kj);
        const [ph, dx] = kk.split('|').map(Number);
        uniq.set(kk, `v${ph}_${dx < 0 ? 'm' + -dx : dx}`);
      }
    }
    body += `  { int py = ${kSplit ? 'yq' : 'y'} * ${sh} + ${ki * dh - pt};\n`;
    body += `    if (py >= 0 && py < ${iH}) {\n`;
    // 열/경계 계산은 ig·kj와 무관 — 행당 1회 (열∈[0,WQ) ⟺ 전역∈[0,W))
    for (const [kk, vn] of uniq) {
      const [, dx] = kk.split('|').map(Number);
      body += `      int c${vn} = x * ${sw} + ${dx};\n`;
      body += `      int q${vn} = clamp(c${vn}, 0, ${iWQ - 1});\n`;
      body += `      float m${vn} = float(c${vn} >= 0 && c${vn} < ${iWQ});\n`;
    }
    if (dw) {
      // depthwise: 입력 레이어 = og (런타임)
      for (const [kk, vn] of uniq) {
        const [ph] = kk.split('|').map(Number);
        body += `      vec4 ${vn} = texelFetch(uIn${ph}, ivec2((og + uInOff) * ${iWQ} + q${vn}, py), 0) * m${vn};\n`;
      }
      for (let kj = 0; kj < kw; kj++) {
        body += `      { vec4 wv = texelFetch(uW, ivec2(${ki * kw + kj}, og), 0);\n`;
        for (let t = 0; t < SPAN; t++) {
          body += `        a${t} += wv * ${uniq.get(at(t, kj))};\n`;
        }
        body += `      }\n`;
      }
    } else {
      // ig는 런타임 루프 — 전개하면 어큐뮬레이터 4개 × 긴 본문으로 레지스터 스필
      // (실측 28.8 → 8.6ms). 가중치 인덱스 b는 전역 ig, 페치 레이어는 소스 로컬.
      let base = 0;
      for (let si = 0; si < S.length; si++) {
        const lo = Math.max(igFrom, base), hiS = Math.min(hi, base + S[si].L);
        if (lo < hiS) {
          body += kSplit
            ? `      for (int ig = max(klo, ${lo}); ig < min(khi, ${hiS}); ++ig) {\n`
            : `      for (int ig = ${lo}; ig < ${hiS}; ++ig) {\n`;
          for (const [kk, vn] of uniq) {
            const [ph] = kk.split('|').map(Number);
            body += `      vec4 ${vn} = texelFetch(${smp(si, ph)}, ivec2((ig - ${base} + ${offU(si)}) * ${iWQ} + q${vn}, py), 0) * m${vn};\n`;
          }
          for (let kj = 0; kj < kw; kj++) {
            body += `      { int b = (ig * ${taps} + ${ki * kw + kj}) * 4;\n`;
            body += `        vec4 w0 = texelFetch(uW, ivec2(b, og), 0), w1 = texelFetch(uW, ivec2(b+1, og), 0),\n`;
            body += `             w2 = texelFetch(uW, ivec2(b+2, og), 0), w3 = texelFetch(uW, ivec2(b+3, og), 0);\n`;
            for (let t = 0; t < SPAN; t++) {
              const v = uniq.get(at(t, kj));
              body += `        a${t} += vec4(dot(w0,${v}), dot(w1,${v}), dot(w2,${v}), dot(w3,${v}));\n`;
            }
            body += `      }\n`;
          }
          body += `      }\n`;
        }
        base += S[si].L;
      }
    }
    body += `    } }\n`;
  }
  // post: 이 conv의 유일 소비자인 binary(add/mul)를 꼬리에 융합.
  // 피연산자 P는 conv보다 먼저 계산된 텐서여야 한다 (load()에서 순서 검사).
  const postE = (t) => post
    ? ` o${t} = r ${post.mode === 'add' ? '+' : '*'} texelFetch(uP${t}, ivec2((uPOff + og) * ${oWQ} + x, y), 0);`
    : ` o${t} = r;`;
  const tail = Array.from({ length: SPAN }, (_, t) => noTail
    ? `  o${t} = a${t};`
    : `  { vec4 v = a${t} + bb; vec4 r = ${ACT[act || 'none']};${postE(t)} }`).join('\n');

  const samplers = srcs
    ? S.map((_, si) => Array.from({ length: SPAN }, (_, p) => `uniform sampler2D uIn${si}_${p};`).join(' ')).join('\n')
    : INS();
  const offs = srcs
    ? S.map((_, si) => `uniform int uInOff${si};`).join(' ')
    : 'uniform int uInOff;';

  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uW, uB;
${samplers}
${post ? `${INS('uP')} uniform int uPOff;` : ''}
uniform bool uHasBias; ${offs} uniform int uOutOff;
${OUTS()}
void main() {${HEAD(oWQ)}
${kSplit ? `  int kc = y / ${kSplit.oH};
  int yq = y - kc * ${kSplit.oH};
  int klo = kc * ${kSplit.CS};
  int khi = min(klo + ${kSplit.CS}, ${totL});\n` : ''}\
  ${Array.from({ length: SPAN }, (_, t) => `vec4 a${t} = vec4(0.0);`).join(' ')}
${noTail ? '' : '  vec4 bb = uHasBias ? texelFetch(uB, ivec2(og, 0), 0) : vec4(0.0);\n'}${body}${tail}
}`;
}

// kSplit 부분합(세로 chunks단) 리듀스 + bias + activation(+post).
const REDUCE_SRC = (act, oWQ, chunks, oH, post = null) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()} uniform sampler2D uB; uniform bool uHasBias; uniform int uInOff, uOutOff;
${post ? `${INS('uP')} uniform int uPOff;` : ''}
${OUTS()}
void main() {${HEAD(oWQ)}
  vec4 bb = uHasBias ? texelFetch(uB, ivec2(og, 0), 0) : vec4(0.0);
${Array.from({ length: SPAN }, (_, t) => `  { vec4 v = bb;
${Array.from({ length: chunks }, (_, c) =>
    `    v += texelFetch(uIn${t}, ivec2(og * ${oWQ} + x, y + ${c * oH}), 0);`).join('\n')}
    vec4 r = ${ACT[act || 'none']};` +
  (post ? ` o${t} = r ${post.mode === 'add' ? '+' : '*'} texelFetch(uP${t}, ivec2((uPOff + og) * ${oWQ} + x, y), 0);`
        : ` o${t} = r;`) + ' }').join('\n')}
}`;

// splitK 누적 결과에 bias+activation(+post binary)만 얹는 마감 (위상 항등 매핑).
const TAIL_SRC = (act, oWQ, post = null) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()} uniform sampler2D uB; uniform bool uHasBias; uniform int uInOff, uOutOff;
${post ? `${INS('uP')} uniform int uPOff;` : ''}
${OUTS()}
void main() {${HEAD(oWQ)}
  vec4 bb = uHasBias ? texelFetch(uB, ivec2(og, 0), 0) : vec4(0.0);
${Array.from({ length: SPAN }, (_, t) =>
  `  { vec4 v = texelFetch(uIn${t}, ivec2((uInOff + og) * ${oWQ} + x, y), 0) + bb; vec4 r = ${ACT[act || 'none']};` +
  (post ? ` o${t} = r ${post.mode === 'add' ? '+' : '*'} texelFetch(uP${t}, ivec2((uPOff + og) * ${oWQ} + x, y), 0);`
        : ` o${t} = r;`) + ' }').join('\n')}
}`;

// GRU 갱신 h' = (1-z)*A + z*B — sub/mul/mul/add 4패스를 mix 1패스로.
const MIX_SRC = (oWQ) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS('uA')} ${INS('uB')} ${INS('uZ')}
uniform int uOffA, uOffB, uOffZ, uOutOff;
${OUTS()}
void main() {${HEAD(oWQ)}
${Array.from({ length: SPAN }, (_, t) =>
  `  o${t} = mix(texelFetch(uA${t}, ivec2((uOffA + og) * ${oWQ} + x, y), 0),
           texelFetch(uB${t}, ivec2((uOffB + og) * ${oWQ} + x, y), 0),
           texelFetch(uZ${t}, ivec2((uOffZ + og) * ${oWQ} + x, y), 0));`).join('\n')}
}`;

const ACT_SRC = (act, oWQ) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()}
${OFFS}
${OUTS()}
void main() {${HEAD(oWQ)}
${Array.from({ length: SPAN }, (_, t) =>
  `  { vec4 v = texelFetch(uIn${t}, ivec2((uInOff + og) * ${oWQ} + x, y), 0); o${t} = ${ACT[act]}; }`).join('\n')}
}`;

const BIN_SRC = (mode, useScalar, scalarFirst, oWQ) => {
  const expr = (t) => {
    const a = `texelFetch(uA${t}, ivec2((uInOff + og) * ${oWQ} + x, y), 0)`;
    const b = useScalar ? 'vec4(uS)' : `texelFetch(uB${t}, ivec2((uInOffB + og) * ${oWQ} + x, y), 0)`;
    const [l, r] = (useScalar && scalarFirst) ? [b, a] : [a, b];
    return mode === 'add' ? `${l} + ${r}` : mode === 'mul' ? `${l} * ${r}` : `${l} - ${r}`;
  };
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS('uA')} ${useScalar ? '' : INS('uB')}
uniform float uS; uniform int uInOffB;
${OFFS}
${OUTS()}
void main() {${HEAD(oWQ)}
${Array.from({ length: SPAN }, (_, t) => `  o${t} = ${expr(t)};`).join('\n')}
}`;
};

// ---- 벡터 텐서 ([1,C,1,1] — SE/gate 경로) ----
// 위상 레이아웃 밖의 단일 텍스처 (L, 1). 전역 평균 → 1×1 conv → broadcast 곱 체인용.

// 전역 평균: 위상 텐서 → 벡터. 출력 텍셀 (og,0) = 레이어 og의 전 픽셀 평균.
// 입력이 저해상(gate는 1/16 스케일)이라 프래그먼트당 수백 페치면 충분.
const GPOOL_SRC = (iWQ, iH) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()}
uniform int uInOff, uOutOff;
out vec4 o;
void main() {
  int og = int(gl_FragCoord.x);
  vec4 acc = vec4(0.0);
  for (int yy = 0; yy < ${iH}; ++yy) {
    for (int xq = 0; xq < ${iWQ}; ++xq) {
${Array.from({ length: SPAN }, (_, p) =>
  `      acc += texelFetch(uIn${p}, ivec2((uInOff + og) * ${iWQ} + xq, yy), 0);`).join('\n')}
    }
  }
  o = acc / float(${iWQ * SPAN * iH});
}`;

// 벡터 1×1 conv (+bias+act) — 행렬×벡터. 가중치 패킹은 일반 conv(taps=1)와 동일.
const VECCONV_SRC = (iL, act) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uIn0, uW, uB;
uniform bool uHasBias; uniform int uInOff, uOutOff;
out vec4 o;
void main() {
  int og = int(gl_FragCoord.x);
  vec4 acc = vec4(0.0);
  for (int ig = 0; ig < ${iL}; ++ig) {
    vec4 xv = texelFetch(uIn0, ivec2(uInOff + ig, 0), 0);
    int b = ig * 4;
    acc += vec4(dot(texelFetch(uW, ivec2(b, og), 0), xv),
                dot(texelFetch(uW, ivec2(b+1, og), 0), xv),
                dot(texelFetch(uW, ivec2(b+2, og), 0), xv),
                dot(texelFetch(uW, ivec2(b+3, og), 0), xv));
  }
  vec4 v = acc + (uHasBias ? texelFetch(uB, ivec2(og, 0), 0) : vec4(0.0));
  o = ${ACT[act || 'none']};
}`;

// broadcast: 위상 텐서 × 벡터 (채널별 게이트 곱 등)
const BCAST_SRC = (mode, vecFirst, oWQ) => {
  const expr = (t) => {
    const a = `texelFetch(uA${t}, ivec2((uInOff + og) * ${oWQ} + x, y), 0)`;
    const b = `texelFetch(uV, ivec2(uVOff + og, 0), 0)`;
    const [l, r] = vecFirst ? [b, a] : [a, b];
    return mode === 'add' ? `${l} + ${r}` : mode === 'mul' ? `${l} * ${r}` : `${l} - ${r}`;
  };
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS('uA')} uniform sampler2D uV;
uniform int uVOff;
${OFFS}
${OUTS()}
void main() {${HEAD(oWQ)}
${Array.from({ length: SPAN }, (_, t) => `  o${t} = ${expr(t)};`).join('\n')}
}`;
};

// 텍셀 내부 성분 복사 (비정렬 Split — 예: RVM 4ch → fgr rgb + pha a)
const CHCOPY_SRC = (srcC, n, oWQ) => {
  const comp = ['r', 'g', 'b', 'a'];
  const parts = Array.from({ length: 4 }, (_, c) =>
    c < n ? `v.${comp[srcC + c]}` : '0.0').join(', ');
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()}
${OFFS}
${OUTS()}
void main() {${HEAD(oWQ)}
${Array.from({ length: SPAN }, (_, t) =>
  `  { vec4 v = texelFetch(uIn${t}, ivec2((uInOff + og) * ${oWQ} + x, y), 0); o${t} = vec4(${parts}); }`).join('\n')}
}`;
};

// Concat/Split — 위상 항등 복사. viewport로 출력 구간을 자르고 uSrcOff로 소스 레이어를 민다.
const COPY_SRC = (oWQ, iWQ) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()} uniform int uSrcOff;
${OFFS}
${OUTS()}
void main() {${HEAD(oWQ)}
${Array.from({ length: SPAN }, (_, t) =>
  `  o${t} = texelFetch(uIn${t}, ivec2((uInOff + og + uSrcOff) * ${iWQ} + x, y), 0);`).join('\n')}
}`;

// Resize(bilinear)만 보간 좌표가 비정수라 위상이 런타임 — if-체인으로 위상 텍스처 선택.
// Ls가 있으면 Concat 융합: 소스 si의 레이어 경계를 og로 갈라 직접 읽는다 (Concat 패스 제거).
const RESIZE_SRC = (iH, iW, oH, oW, halfPixel, Ls = null) => {
  const iWQ = iW / SPAN, oWQ = oW / SPAN;
  const nS = Ls ? Ls.length : 1;
  const ins = Ls
    ? Ls.map((_, si) => Array.from({ length: SPAN }, (_, p) => `uniform sampler2D uIn${si}_${p};`).join(' ')).join('\n')
    : INS();
  const offs = Ls ? Ls.map((_, si) => `uniform int uInOff${si};`).join(' ') + ' uniform int uOutOff;' : OFFS;
  const chain = (si) => Array.from({ length: SPAN }, (_, p) =>
    `  ${p < SPAN - 1 ? `if (ph == ${p}) ` : ''}return texelFetch(${Ls ? `uIn${si}_${p}` : `uIn${p}`}, pos, 0);`)
    .filter((_, p) => p < SPAN).join('\n');
  // 소스 선택은 og 기준 (프래그먼트 간 일관 분기)
  let sel = '';
  if (Ls) {
    let base = 0;
    for (let si = 0; si < nS; si++) {
      sel += si < nS - 1
        ? `  ${si ? 'else ' : ''}if (og < ${base + Ls[si]}) { si = ${si}; base = (og - ${base} + uInOff${si}) * ${iWQ}; }\n`
        : `  else { si = ${si}; base = (og - ${base} + uInOff${si}) * ${iWQ}; }\n`;
      base += Ls[si];
    }
  }
  const one = (t) => `
  { float px = float(x * ${SPAN} + ${t}) + 0.5;
    vec2 p = ${halfPixel ? `vec2(px, float(y) + 0.5) * sc - 0.5`
                         : `(vec2(px, float(y) + 0.5) - 0.5) * sc`};
    vec2 f = floor(p), tt = clamp(p - f, 0.0, 1.0);
    int gx0 = int(clamp(f.x, 0.0, ${iW - 1}.0)), gx1 = int(clamp(f.x + 1.0, 0.0, ${iW - 1}.0));
    int gy0 = int(clamp(f.y, 0.0, ${iH - 1}.0)), gy1 = int(clamp(f.y + 1.0, 0.0, ${iH - 1}.0));
    vec4 a = F(si, base, gx0, gy0), b = F(si, base, gx1, gy0);
    vec4 c = F(si, base, gx0, gy1), d = F(si, base, gx1, gy1);
    o${t} = mix(mix(a, b, tt.x), mix(c, d, tt.x), tt.y); }`;
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${ins}
${offs}
${OUTS()}
vec4 F(int si, int base, int gx, int yy) {
  ivec2 pos = ivec2(base + gx / ${SPAN}, yy);
  int ph = gx % ${SPAN};
${Ls && nS > 1
  ? Ls.map((_, si) => `  ${si < nS - 1 ? `if (si == ${si}) ` : ''}{\n${chain(si)}\n  }`).join('\n')
  : chain(0)}
}
void main() {${HEAD(oWQ)}
  vec2 sc = vec2(${(iW / oW).toFixed(8)}, ${(iH / oH).toFixed(8)});
  int si = 0;
  ${Ls ? 'int base;' : `int base = (uInOff + og) * ${iWQ};`}
${sel}${Array.from({ length: SPAN }, (_, t) => one(t)).join('\n')}
}`;
};

// AvgPool(k=s) — conv와 같은 이유로 위상이 정적. 탭을 전개해 위상별로 모은다.
const AVGPOOL_SRC = (kh, kw, sh, sw, iH, iW, oWQ) => {
  const iWQ = iW / SPAN;
  let body = '';
  for (let ki = 0; ki < kh; ki++) {
    const uniq = new Map();
    for (let t = 0; t < SPAN; t++) {
      for (let j = 0; j < kw; j++) {
        const c = t * sw + j;
        uniq.set(`${phaseOf(c)}|${dxOf(c)}`, `v${phaseOf(c)}_${dxOf(c)}`);
      }
    }
    body += `  { int py = min(y * ${sh} + ${ki}, ${iH - 1});\n`;
    for (const [kk, vn] of uniq) {
      const [ph, dx] = kk.split('|').map(Number);
      body += `    vec4 ${vn} = texelFetch(uIn${ph}, ivec2(base + min(x * ${sw} + ${dx}, ${iWQ - 1}), py), 0);\n`;
    }
    for (let t = 0; t < SPAN; t++) {
      const terms = [];
      for (let j = 0; j < kw; j++) {
        const c = t * sw + j;
        terms.push(uniq.get(`${phaseOf(c)}|${dxOf(c)}`));
      }
      body += `    a${t} += ${terms.join(' + ')};\n`;
    }
    body += `  }\n`;
  }
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
${INS()}
${OFFS}
${OUTS()}
void main() {${HEAD(oWQ)}
  int base = (uInOff + og) * ${iWQ};
  ${Array.from({ length: SPAN }, (_, t) => `vec4 a${t} = vec4(0.0);`).join(' ')}
${body}${Array.from({ length: SPAN }, (_, t) =>
  `  o${t} = a${t} / float(${kh * kw});`).join('\n')}
}`;
};

export class Webgl2Engine {
  constructor(gl, opts = {}) {
    this.gl = gl;
    this.debug = !!opts.debug;
    this.noDraw = !!opts.noDraw;
    this.splitView = opts.splitView !== false;
    this.fuseConcat = opts.fuseConcat !== false;
    this.fuseMix = opts.fuseMix !== false;       // GRU h' 4패스 → mix 1패스
    this.fusePost = opts.fusePost !== false;     // conv 유일 소비 binary를 꼬리에
    this.inval = opts.inval !== false;           // 전체 덮어쓰기 draw 전 타일 load 생략
    this.invAtt = Array.from({ length: SPAN }, (_, p) => 0x8CE0 + p);  // COLOR_ATTACHMENT0+p
    this.splitK = opts.splitK ?? 12;   // 루프 코드젠 후 실측: 6→7.06 / 12→6.87 / 18→6.97 / 0→6.90
    this.cur = { fbo: null, prog: null, tex: new Array(16).fill(null), vp: '' };
    this.uvals = new Map();
    this.progs = new Map();
    this.tex = new Map();
    this.fbos = new Map();
    this.vao = gl.createVertexArray();
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float 없음');
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < SPAN) {
      throw new Error(`MAX_DRAW_BUFFERS ${gl.getParameter(gl.MAX_DRAW_BUFFERS)} < SPAN ${SPAN}`);
    }
    this.stats = { draws: 0 };
    this.notes = [`위상 SPAN${SPAN}`, '활성 fp32', `MAX_TEXTURE_SIZE ${this.maxTex}`];
    if (this.debug) this.notes.push('⚠ debug=on (draw마다 동기화 — 타이밍 무효)');
  }

  _prog(key, src) {
    if (this.progs.has(key)) return this.progs.get(key);
    const gl = this.gl;
    const mk = (ty, s) => {
      const sh = gl.createShader(ty);
      gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(`컴파일 실패 [${key}] ${gl.getShaderInfoLog(sh)}`);
      }
      return sh;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, src));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(`링크 실패 [${key}] ${gl.getProgramInfoLog(p)}`);
    }
    const u = {};
    for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
      const n = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
      u[n] = gl.getUniformLocation(p, n);
    }
    const rec = { p, u };
    this.progs.set(key, rec);
    return rec;
  }

  // 유니폼은 프로그램별 영속 상태 — 같은 값 재설정은 생략 (draw당 CPU 비용 절감).
  // 캐시 키 = WebGLUniformLocation (프로그램마다 고유).
  _setU(loc, v) {
    if (!loc) return;
    if (this.uvals.get(loc) === v) return;
    this.gl.uniform1i(loc, v);
    this.uvals.set(loc, v);
  }

  _mkTexs(TWQ, H) {
    const gl = this.gl;
    const texs = [];
    for (let p = 0; p < SPAN; p++) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, TWQ, H);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST],
                            [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
                            [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) {
        gl.texParameteri(gl.TEXTURE_2D, k, v);
      }
      texs.push(tex);
    }
    return texs;
  }

  _alloc(name, shape) {
    if (this.tex.has(name)) return this.tex.get(name);
    const [, C, H, W] = shape;
    if (H === 1 && W === 1) {
      // 벡터 텐서 [1,C,1,1] — 위상 밖 단일 텍스처 (L, 1)
      const L = Math.ceil(C / 4);
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, L, 1);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      const rec = { vec: true, texs: [tex], W: 1, WQ: 1, H: 1, L, C, TWQ: L, name, off: 0 };
      this.tex.set(name, rec);
      return rec;
    }
    if (W % SPAN) throw new Error(`${name} W=${W}가 SPAN=${SPAN}으로 안 나뉨`);
    const L = Math.ceil(C / 4);
    const WQ = W / SPAN;
    const TWQ = WQ * L;
    if (TWQ > this.maxTex || H > this.maxTex) {
      throw new Error(`${name} 타일 ${TWQ}x${H} > MAX_TEXTURE_SIZE ${this.maxTex}`);
    }
    const rec = { texs: this._mkTexs(TWQ, H), W, WQ, H, L, C, TWQ, name, off: 0 };
    this.tex.set(name, rec);
    return rec;
  }

  _tex2d(data, w, h) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  }

  _scratch(o, mulH = 1) {
    const key = `${o.TWQ}x${o.H * mulH}`;
    let r = (this.scratch = this.scratch || new Map()).get(key);
    if (r) return r;
    r = { texs: this._mkTexs(o.TWQ, o.H * mulH), W: o.W, WQ: o.WQ, H: o.H * mulH,
          L: o.L, C: o.C, TWQ: o.TWQ, name: `__scratch_${key}`, off: 0 };
    this.scratch.set(key, r);
    return r;
  }

  _fbo(rec) {
    let f = this.fbos.get(rec.name);
    if (f) return f;
    const gl = this.gl;
    f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    const atts = [];
    const n = rec.vec ? 1 : SPAN;
    for (let p = 0; p < n; p++) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + p,
                              gl.TEXTURE_2D, rec.texs[p], 0);
      atts.push(gl.COLOR_ATTACHMENT0 + p);
    }
    gl.drawBuffers(atts);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`FBO 0x${st.toString(16)} (${rec.name})`);
    this.fbos.set(rec.name, f);
    this.cur.fbo = f;
    return f;
  }

  async load(planUrl, weightsUrl) {
    // plan과 가중치는 독립 — 병렬 fetch (직렬이면 RTT가 더해진다)
    const [plan, wbuf] = await Promise.all([
      fetch(planUrl).then((r) => r.json()),
      fetch(weightsUrl).then((r) => r.arrayBuffer()),
    ]);
    this.plan = plan;
    // 가중치 참조별 디코드 — --fp16 패킹이면 std conv만 fp16 저장(ref.f16),
    // dw/bias는 fp32 유지 (BN 접힘 다이내믹 레인지 때문). 연산은 항상 fp32.
    const u32 = new Uint32Array(1);
    const f32 = new Float32Array(u32.buffer);
    const wslice = (ref) => {
      if (!ref.f16) return new Float32Array(wbuf, ref.off, ref.len);
      const h = new Uint16Array(wbuf, ref.off, ref.len);
      const out = new Float32Array(ref.len);
      for (let i = 0; i < h.length; i++) {
        const x = h[i];
        const sgn = (x & 0x8000) << 16;
        let e = (x >> 10) & 0x1f;
        let f = x & 0x3ff;
        if (e === 0) {
          if (f === 0) { u32[0] = sgn; }
          else {                                  // subnormal 정규화
            e = 113;
            while (!(f & 0x400)) { f <<= 1; e--; }
            u32[0] = sgn | (e << 23) | ((f & 0x3ff) << 13);
          }
        } else if (e === 31) {
          u32[0] = sgn | 0x7f800000 | (f << 13);
        } else {
          u32[0] = sgn | ((e + 112) << 23) | (f << 13);
        }
        out[i] = f32[0];
      }
      return out;
    };
    // Transpose 출력은 별칭이라 할당하지 않는다 (segment는 NHWC shape이라 W%SPAN도 안 맞음)
    const alias = new Set(this.plan.ops
      .filter((o) => o.type === 'Transpose' || o.type === 'alias')
      .map((o) => o.out[0]));
    for (const [n, t] of Object.entries(this.plan.tensors)) {
      if (!alias.has(n)) this._alloc(n, t.shape);
    }
    for (const io of [...this.plan.inputs, ...this.plan.outputs]) {
      if (!alias.has(io.name)) this._alloc(io.name, io.shape);
    }
    // 상수 벡터 텐서 (ImageNet 정규화 등) — 벡터 텍스처로 업로드
    for (const [cn, c] of Object.entries(this.plan.consts || {})) {
      const rec = this._alloc(cn, c.shape);
      const buf = new Float32Array(rec.L * 4);
      c.values.forEach((v, i) => { buf[i] = v; });
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, rec.texs[0]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, rec.L, 1, gl.RGBA, gl.FLOAT, buf);
    }
    for (const op of this.plan.ops) {
      if (op.type !== 'Conv') continue;
      op._wt = this._tex2d(wslice(op.w),
                           op.wtex[0], op.wtex[1]);
      if (op.b) {
        op._bt = this._tex2d(wslice(op.b),
                             op.btex[0], op.btex[1]);
      }
    }
    this.dummy = this._tex2d(new Float32Array(4), 1, 1);

    const uses = new Map();
    for (const op of this.plan.ops) {
      for (const t of op.in) uses.set(t, (uses.get(t) || 0) + 1);
    }
    for (const t of this.plan.outputs.map((o) => o.name)) {
      uses.set(t, (uses.get(t) || 0) + 1);
    }
    const byOut = new Map();
    for (const o of this.plan.ops) for (const t of o.out) byOut.set(t, o);
    const idxOf = new Map(this.plan.ops.map((o, i) => [o, i]));

    // Concat → conv/Resize 융합: Concat 출력의 유일 소비자가 소스를 직접 읽는다.
    // 샘플러 예산: 소스 3 × 위상 4 + uW + uB = 14 ≤ 16. 4소스는 불가(18)지만 plan에 없음.
    if (this.fuseConcat) {
      let n = 0;
      for (const op of this.plan.ops) {
        if (op.type !== 'Conv' && op.type !== 'Resize') continue;
        if (op.type === 'Conv' && op.dw) continue;
        const prod = byOut.get(op.in[0]);
        if (!prod || prod.type !== 'Concat') continue;
        if (uses.get(op.in[0]) !== 1) continue;      // 다른 소비자가 있으면 불가
        if (prod.in.length * SPAN + 2 > 16) continue; // 샘플러 유닛 한계
        op._srcs = prod.in.slice();
        prod._skip = true;
        n++;
      }
      this.notes.push(`Concat 융합 ${n}개`);
    }

    // GRU 갱신 mix 융합: add(mul(1-z, A), mul(z, B)) → h' = mix(A, B, z) 1패스
    if (this.fuseMix) {
      let n = 0;
      for (const op of this.plan.ops) {
        if (op.type !== 'binary' || op.mode !== 'add' || op.scalar !== undefined) continue;
        const m1 = byOut.get(op.in[0]), m2 = byOut.get(op.in[1]);
        if (!m1 || !m2 || m1.type !== 'binary' || m2.type !== 'binary') continue;
        if (m1.mode !== 'mul' || m2.mode !== 'mul') continue;
        if (uses.get(op.in[0]) !== 1 || uses.get(op.in[1]) !== 1) continue;
        outer:
        for (const [ma, mb] of [[m1, m2], [m2, m1]]) {
          for (let si = 0; si < 2; si++) {
            const s = byOut.get(ma.in[si]);
            if (!s || s.type !== 'binary' || s.mode !== 'sub' || s.scalar !== 1
                || !s.scalar_first || uses.get(ma.in[si]) !== 1) continue;
            const z = s.in[0];
            const zi = mb.in.indexOf(z);
            if (zi < 0) continue;
            op._mix = { A: ma.in[1 - si], B: mb.in[1 - zi], z };
            s._skip = ma._skip = mb._skip = true;
            n++;
            break outer;
          }
        }
      }
      this.notes.push(`mix 융합 ${n}개`);
    }

    // conv 꼬리 binary 융합: conv 출력의 유일 소비자가 add/mul이고 다른 피연산자가
    // conv보다 먼저 계산돼 있으면 꼬리에서 바로 연산 (패스 1개 제거).
    // ⚠ 피연산자가 conv 뒤에 생산되면 쓰기 전 읽기 — 순서 검사 필수.
    if (this.fusePost) {
      let n = 0;
      for (const op of this.plan.ops) {
        if (op.type !== 'binary' || op._skip || op._mix) continue;
        if ((op.mode !== 'add' && op.mode !== 'mul') || op.scalar !== undefined) continue;
        const shOf = (t) => (this.plan.tensors[t] || {}).shape;
        for (const [ci, oi] of [[0, 1], [1, 0]]) {
          const prod = byOut.get(op.in[ci]);
          if (!prod || prod.type !== 'Conv' || prod._skip || prod._post) continue;
          if (uses.get(op.in[ci]) !== 1) continue;
          if ((prod._srcs ? prod._srcs.length : 1) > 2) continue;   // 샘플러 예산
          // 벡터 텐서가 끼면 (SE/gate·상수벡터 경로) 꼬리 융합 불가 — bcast 셰이더가 담당
          const so = shOf(op.in[oi]), sc2 = shOf(prod.in[0]);
          if (so && so[2] === 1 && so[3] === 1) continue;
          if (sc2 && sc2[2] === 1 && sc2[3] === 1) continue;
          if (this.plan.consts && this.plan.consts[op.in[oi]]) continue;
          const other = byOut.get(op.in[oi]);
          if (other && idxOf.get(other) > idxOf.get(prod)) continue; // 순서 위반
          prod._post = { mode: op.mode, src: op.in[oi], out: op.out[0] };
          op._skip = true;
          n++;
          break;
        }
      }
      this.notes.push(`conv꼬리 융합 ${n}개`);
    }
    this.notes.push(`텐서 ${this.tex.size} / 패스 ${this.plan.ops.length}`);
    return this;
  }

  upload(name, data) {
    const gl = this.gl, r = this.tex.get(name);
    const plane = r.WQ * r.H;
    const buf = new Float32Array(plane * 4);
    for (let p = 0; p < SPAN; p++) {
      gl.bindTexture(gl.TEXTURE_2D, r.texs[p]);
      for (let l = 0; l < r.L; l++) {
        buf.fill(0);
        for (let c = 0; c < 4; c++) {
          const ch = l * 4 + c;
          if (ch >= r.C) break;
          const src = ch * r.W * r.H;
          for (let y = 0; y < r.H; y++) {
            for (let xq = 0; xq < r.WQ; xq++) {
              buf[(y * r.WQ + xq) * 4 + c] = data[src + y * r.W + xq * SPAN + p];
            }
          }
        }
        gl.texSubImage2D(gl.TEXTURE_2D, 0, l * r.WQ, 0, r.WQ, r.H, gl.RGBA, gl.FLOAT, buf);
      }
    }
  }

  download(name) {
    const gl = this.gl, r = this.tex.get(name);
    const out = new Float32Array(r.C * r.W * r.H);
    const raw = new Float32Array(r.TWQ * r.H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo(r));
    this.cur.fbo = null;                       // readBuffer 상태를 건드리므로 캐시 무효화
    for (let p = 0; p < SPAN; p++) {
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + p);
      gl.readPixels(0, 0, r.TWQ, r.H, gl.RGBA, gl.FLOAT, raw);
      for (let l = 0; l < r.L; l++) {
        for (let c = 0; c < 4; c++) {
          const ch = l * 4 + c;
          if (ch >= r.C) break;
          const dst = ch * r.W * r.H;
          for (let y = 0; y < r.H; y++) {
            for (let xq = 0; xq < r.WQ; xq++) {
              out[dst + y * r.W + xq * SPAN + p] =
                raw[(y * r.TWQ + (l + r.off) * r.WQ + xq) * 4 + c];
            }
          }
        }
      }
    }
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    return out;
  }

  _draw(o, prog, setup, xOff = 0, w = 0, keep = false) {
    const gl = this.gl;
    const fbo = this._fbo(o);
    if (this.cur.fbo !== fbo) { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); this.cur.fbo = fbo; }
    // TBDR은 패스마다 어태치먼트를 타일로 load한다. 전체를 덮어쓰는 draw면 이전
    // 내용이 필요 없다고 알려 load를 생략시킨다 (keep=블렌딩 누적/부분 뷰포트는 예외).
    if (!keep && !xOff && !w && this.inval) {
      gl.invalidateFramebuffer(gl.FRAMEBUFFER, o.vec ? [0x8CE0] : this.invAtt);
    }
    const vp = `${xOff}|${w || o.TWQ}|${o.H}`;
    if (this.cur.vp !== vp) { gl.viewport(xOff, 0, w || o.TWQ, o.H); this.cur.vp = vp; }
    if (this.cur.prog !== prog.p) { gl.useProgram(prog.p); this.cur.prog = prog.p; }
    this._setU(prog.u.uInOff, 0);
    this._setU(prog.u.uInOffB, 0);
    this._setU(prog.u.uOutOff, o.off || 0);
    setup(prog);
    if (!this.noDraw) gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.stats.draws++;
    if (this.debug) {
      const e = gl.getError();
      if (e) throw new Error(`GL ${e}`);
    }
  }

  _bind(unit, tex, loc) {
    const gl = this.gl;
    if (this.cur.tex[unit] !== tex) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      this.cur.tex[unit] = tex;
    }
    this._setU(loc, unit);        // 샘플러 유닛도 프로그램별 영속 — 값 같으면 생략
  }

  // 위상 텍스처 4개를 unit0..3(또는 base..base+3)에 바인딩
  _bindPhases(rec, prog, pfx = 'uIn', base = 0) {
    for (let p = 0; p < SPAN; p++) this._bind(base + p, rec.texs[p], prog.u[`${pfx}${p}`]);
  }

  run(limit = 0) {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    this.stats.draws = 0;
    this.cur = { fbo: null, prog: null, tex: new Array(16).fill(null), vp: '' };
    const ops = this.plan.ops;
    const n = limit || ops.length;
    for (let i = 0; i < n; i++) {
      if (ops[i]._skip) continue;
      try {
        this._exec(ops[i]);
      } catch (e) {
        throw new Error(`op[${i}] ${ops[i].type} ${ops[i].out[0]} — ${e.message}`);
      }
    }
  }

  _exec(op) {
    const gl = this.gl;
    const T = (n) => this.tex.get(n);
    switch (op.type) {
      case 'Conv': {
        const o = T(op.out[0]);
        const iv = T(op.in[0]);
        if (iv.vec) {
          // 벡터 1×1 conv (SE/gate) — 행렬×벡터, 단일 어태치먼트
          if (op.k[0] !== 1 || op.k[1] !== 1 || op.dw) {
            throw new Error(`벡터 입력 conv는 1x1만 지원 (k=${op.k}, dw=${op.dw})`);
          }
          const prog = this._prog(`vecconv|${iv.L}|${op.act}`, VECCONV_SRC(iv.L, op.act));
          this._draw(o, prog, (p) => {
            this._bind(0, iv.texs[0], p.u.uIn0);
            this._setU(p.u.uInOff, iv.off);
            this._bind(1, op._wt, p.u.uW);
            this._bind(2, op._bt || this.dummy, p.u.uB);
            this._setU(p.u.uHasBias, op._bt ? 1 : 0);
          });
          break;
        }
        const recs = op._srcs ? op._srcs.map(T) : [T(op.in[0])];
        const i0 = recs[0];
        const srcs = op._srcs ? recs.map((r) => ({ L: r.L })) : null;
        const srcKey = srcs ? recs.map((r) => r.L).join(',') : 'x';
        // 입력 소스들 위상 바인딩 + 레이어 오프셋 (융합 시 소스별, 아니면 단일)
        const bindIns = (p) => {
          if (srcs) {
            recs.forEach((r, si) => {
              this._bindPhases(r, p, `uIn${si}_`, si * SPAN);
              this._setU(p.u[`uInOff${si}`], r.off);
            });
          } else {
            this._bindPhases(i0, p);
            this._setU(p.u.uInOff, i0.off);
          }
        };
        const wUnit = (srcs ? recs.length : 1) * SPAN;
        const totL = srcs ? recs.reduce((a, r) => a + r.L, 0) : i0.L;
        const CH = this.splitK;
        // 프래그먼트 기아 conv (깊은 저해상 구간): 청크 부분합을 세로로 쌓아 병렬화.
        // blend 누적 splitK는 순차 패스라 기아를 못 푼다.
        const frags = o.TWQ * o.H;
        if (!op.dw && frags < 16384 && totL > 4) {
          let chunks = Math.min(Math.ceil(32768 / frags), Math.ceil(totL / 2), 8);
          const CS = Math.ceil(totL / chunks);
          chunks = Math.ceil(totL / CS);
          if (chunks >= 2) {
            const acc = this._scratch(o, chunks);
            const kk = `convp|${op.k}|${op.s}|${op.pad}|${op.dil}|${i0.WQ}|${i0.H}|${totL}|${o.WQ}|${CS}x${chunks}|${srcKey}`;
            const pk = this._prog(kk, convSrc({
              dw: false, k: op.k, s: op.s, pad: op.pad, act: 'none', dil: op.dil,
              iWQ: i0.WQ, iH: i0.H, iL: i0.L, oWQ: o.WQ, srcs,
              noTail: true, kSplit: { CS, chunks, oH: o.H },
            }));
            this._draw(acc, pk, (p) => {
              bindIns(p);
              this._bind(wUnit, op._wt, p.u.uW);
            });
            const dstR = op._post ? T(op._post.out) : o;
            const rp = this._prog(`reduce|${op.act}|${o.WQ}|${chunks}|${o.H}|${op._post ? op._post.mode : ''}`,
                                  REDUCE_SRC(op.act, o.WQ, chunks, o.H, op._post));
            this._draw(dstR, rp, (p) => {
              this._bindPhases(acc, p);
              this._bind(SPAN, op._bt || this.dummy, p.u.uB);
              this._setU(p.u.uHasBias, op._bt ? 1 : 0);
              if (op._post) {
                const P = T(op._post.src);
                this._bindPhases(P, p, 'uP', SPAN + 1);
                this._setU(p.u.uPOff, P.off);
              }
            });
            break;
          }
        }
        if (CH && !op.dw && totL > CH) {
          const acc = this._scratch(o);
          // 청크별 부분합을 블렌딩으로 누적 → 마지막에 bias+activation 한 번
          for (let from = 0; from < totL; from += CH) {
            const to = Math.min(from + CH, totL);
            const kk = `convk|${op.k}|${op.s}|${op.pad}|${op.dil}|${i0.WQ}|${i0.H}|${totL}|${o.WQ}|${from}-${to}|${srcKey}`;
            const pk = this._prog(kk, convSrc({
              dw: false, k: op.k, s: op.s, pad: op.pad, act: 'none', dil: op.dil,
              iWQ: i0.WQ, iH: i0.H, iL: i0.L, oWQ: o.WQ, srcs,
              igFrom: from, igTo: to, noTail: true,
            }));
            if (from === 0) gl.disable(gl.BLEND);
            else { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); }
            // from>0은 블렌딩 누적 — 부분합을 load해야 하므로 invalidate 금지
            this._draw(acc, pk, (p) => {
              bindIns(p);
              this._bind(wUnit, op._wt, p.u.uW);
            }, 0, 0, from > 0);
          }
          gl.disable(gl.BLEND);
          const dstT = op._post ? T(op._post.out) : o;
          const tp = this._prog(`tail|${op.act}|${o.WQ}|${op._post ? op._post.mode : ''}`,
                                TAIL_SRC(op.act, o.WQ, op._post));
          this._draw(dstT, tp, (p) => {
            this._bindPhases(acc, p);
            this._bind(SPAN, op._bt || this.dummy, p.u.uB);
            this._setU(p.u.uHasBias, op._bt ? 1 : 0);
            if (op._post) {
              const P = T(op._post.src);
              this._bindPhases(P, p, 'uP', SPAN + 1);
              this._setU(p.u.uPOff, P.off);
            }
          });
          break;
        }
        const key = `conv|${op.dw}|${op.k}|${op.s}|${op.pad}|${op.dil}|${op.act}|${i0.WQ}|${i0.H}|${i0.L}|${o.WQ}|${srcKey}|${op._post ? op._post.mode : ''}`;
        const prog = this._prog(key, convSrc({
          dw: op.dw, k: op.k, s: op.s, pad: op.pad, act: op.act, dil: op.dil,
          iWQ: i0.WQ, iH: i0.H, iL: i0.L, oWQ: o.WQ, srcs, post: op._post,
        }));
        const dst = op._post ? T(op._post.out) : o;
        this._draw(dst, prog, (p) => {
          bindIns(p);
          this._bind(wUnit, op._wt, p.u.uW);
          this._bind(wUnit + 1, op._bt || this.dummy, p.u.uB);
          this._setU(p.u.uHasBias, op._bt ? 1 : 0);
          if (op._post) {
            const P = T(op._post.src);
            this._bindPhases(P, p, 'uP', wUnit + 2);
            this._setU(p.u.uPOff, P.off);
          }
        });
        break;
      }
      case 'act': {
        const i = T(op.in[0]), o = T(op.out[0]);
        const prog = this._prog(`act|${op.act}|${o.WQ}`, ACT_SRC(op.act, o.WQ));
        this._draw(o, prog, (p) => {
          this._bindPhases(i, p);
          this._setU(p.u.uInOff, i.off);
        });
        break;
      }
      case 'binary': {
        const o = T(op.out[0]);
        if (op._mix) {
          const A = T(op._mix.A), B = T(op._mix.B), Z = T(op._mix.z);
          const prog = this._prog(`mix|${o.WQ}`, MIX_SRC(o.WQ));
          this._draw(o, prog, (p) => {
            this._bindPhases(A, p, 'uA', 0);
            this._bindPhases(B, p, 'uB', SPAN);
            this._bindPhases(Z, p, 'uZ', SPAN * 2);
            this._setU(p.u.uOffA, A.off);
            this._setU(p.u.uOffB, B.off);
            this._setU(p.u.uOffZ, Z.off);
          });
          break;
        }
        const sc = op.scalar !== undefined;
        if (!sc) {
          const A0 = T(op.in[0]), B0 = T(op.in[1]);
          if (A0.vec !== B0.vec) {
            // broadcast: 위상 텐서 (op) 벡터 — 채널별 게이트
            const tR = A0.vec ? B0 : A0, vR = A0.vec ? A0 : B0;
            const prog = this._prog(`bcast|${op.mode}|${A0.vec}|${o.WQ}`,
                                    BCAST_SRC(op.mode, A0.vec, o.WQ));
            this._draw(o, prog, (p) => {
              this._bindPhases(tR, p, 'uA', 0);
              this._setU(p.u.uInOff, tR.off);
              this._bind(SPAN, vR.texs[0], p.u.uV);
              this._setU(p.u.uVOff, vR.off);
            });
            break;
          }
        }
        const prog = this._prog(`bin|${op.mode}|${sc}|${!!op.scalar_first}|${o.WQ}`,
                                BIN_SRC(op.mode, sc, !!op.scalar_first, o.WQ));
        const A = T(op.in[0]);
        this._draw(o, prog, (p) => {
          this._bindPhases(A, p, 'uA', 0);
          this._setU(p.u.uInOff, A.off);
          if (sc) gl.uniform1f(p.u.uS, op.scalar);
          else {
            const B = T(op.in[1]);
            this._bindPhases(B, p, 'uB', SPAN);
            this._setU(p.u.uInOffB, B.off);
          }
        });
        break;
      }
      case 'Concat': {
        const o = T(op.out[0]);
        let off = 0;
        for (const src of op.in) {
          const s = T(src);
          const prog = this._prog(`copy|${o.WQ}|${s.WQ}`, COPY_SRC(o.WQ, s.WQ));
          // viewport로 출력의 [off*WQ, off*WQ + s.L*WQ) 구간만 그린다.
          // gl_FragCoord는 뷰포트 오프셋을 포함하므로 og도 그만큼 크다 → uSrcOff로 되돌린다.
          this._draw(o, prog, (p) => {
            this._bindPhases(s, p);
            this._setU(p.u.uInOff, s.off);
            this._setU(p.u.uSrcOff, -off);
          }, off * o.WQ, s.L * o.WQ);
          off += s.L;
        }
        break;
      }
      case 'Split': {
        const i = T(op.in[0]);
        let off = 0;
        if (this.splitView) {
          // 복사 없이 뷰로 — 출력은 입력의 연속 레이어 구간 (위상 텍스처 공유)
          for (const name of op.out) {
            const o = this.plan.tensors[name];
            const C = o.shape[1], L = Math.ceil(C / 4);
            this.tex.set(name, { texs: i.texs, W: i.W, WQ: i.WQ, H: i.H, TWQ: i.TWQ,
                                 name, C, L, off: i.off + off });
            off += L;
          }
        } else {
          for (const name of op.out) {
            const o = T(name);
            const prog = this._prog(`copy|${o.WQ}|${i.WQ}`, COPY_SRC(o.WQ, i.WQ));
            this._draw(o, prog, (p) => {
              this._bindPhases(i, p);
              this._setU(p.u.uInOff, i.off);
              this._setU(p.u.uSrcOff, off);
            });
            off += o.L;
          }
        }
        break;
      }
      case 'Resize': {
        const o = T(op.out[0]);
        if (op._srcs) {
          const recs = op._srcs.map(T);
          const i0 = recs[0];
          const Ls = recs.map((r) => r.L);
          const prog = this._prog(`rsc|${i0.H}x${i0.W}->${o.H}x${o.W}|${op.ctm}|${Ls.join(',')}`,
            RESIZE_SRC(i0.H, i0.W, o.H, o.W, op.ctm === 'half_pixel', Ls));
          this._draw(o, prog, (p) => {
            recs.forEach((r, si) => {
              this._bindPhases(r, p, `uIn${si}_`, si * SPAN);
              this._setU(p.u[`uInOff${si}`], r.off);
            });
          });
          break;
        }
        const i = T(op.in[0]);
        const prog = this._prog(`rs|${i.H}x${i.W}->${o.H}x${o.W}|${op.ctm}`,
          RESIZE_SRC(i.H, i.W, o.H, o.W, op.ctm === 'half_pixel'));
        this._draw(o, prog, (p) => {
          this._bindPhases(i, p);
          this._setU(p.u.uInOff, i.off);
        });
        break;
      }
      case 'gpool': {
        // 전역 평균: 위상 텐서 → 벡터
        const i = T(op.in[0]), o = T(op.out[0]);
        const prog = this._prog(`gpool|${i.WQ}x${i.H}`, GPOOL_SRC(i.WQ, i.H));
        this._draw(o, prog, (p) => {
          this._bindPhases(i, p);
          this._setU(p.u.uInOff, i.off);
        });
        break;
      }
      case 'AveragePool': {
        const i = T(op.in[0]), o = T(op.out[0]);
        const prog = this._prog(`ap|${op.k}|${op.s}|${i.H}x${i.W}|${o.WQ}`,
          AVGPOOL_SRC(op.k[0], op.k[1], op.s[0], op.s[1], i.H, i.W, o.WQ));
        this._draw(o, prog, (p) => {
          this._bindPhases(i, p);
          this._setU(p.u.uInOff, i.off);
        });
        break;
      }
      case 'chcopy': {
        // 텍셀 내부 성분 분리 (비정렬 Split)
        const i = T(op.in[0]), o = T(op.out[0]);
        const prog = this._prog(`chc|${op.srcC}|${op.n}|${o.WQ}`,
                                CHCOPY_SRC(op.srcC, op.n, o.WQ));
        this._draw(o, prog, (p) => {
          this._bindPhases(i, p);
          this._setU(p.u.uInOff, i.off);
        });
        break;
      }
      case 'alias':
      case 'Transpose':
        this.tex.set(op.out[0], T(op.in[0]));
        break;
      default:
        throw new Error(`미구현 op: ${op.type}`);
    }
  }
}
