// Analiza kuta koljena — isti algoritam kao knee_angle.py (v2), bez ovisnosti o pregledniku,
// da se može provjeriti u Nodeu nad CSV-ovima iz Pythona (analysis.test.mjs).
//
// Uzorak (sample) = { t, angL, angR, visL, visR, zL, zR }; kut je kuk–koljeno–gležanj
// (180° = ispruženo), NaN kad zglobovi nisu dovoljno vidljivi. Fleksija = 180 − kut.

export const LANDMARKS = {
  left: { hip: 23, knee: 25, ankle: 27 },
  right: { hip: 24, knee: 26, ankle: 28 },
};

export function angleDegrees(a, b, c) {
  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const bcx = c.x - b.x;
  const bcy = c.y - b.y;
  const denominator = Math.hypot(bax, bay) * Math.hypot(bcx, bcy);
  if (denominator === 0) return null;
  const cosine = Math.max(-1, Math.min(1, (bax * bcx + bay * bcy) / denominator));
  return (Math.acos(cosine) * 180) / Math.PI;
}

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Iz landmarkova jednog framea (normalizirane koordinate) složi uzorak za obje noge. */
export function legSample(landmarks, width, height, t, minVisibility = 0.5) {
  const sample = { t, angL: NaN, angR: NaN, visL: 0, visR: 0, zL: NaN, zR: NaN };
  if (!landmarks) return sample;
  for (const [side, suffix] of [["left", "L"], ["right", "R"]]) {
    const joints = LANDMARKS[side];
    const points = [joints.hip, joints.knee, joints.ankle].map((index) => landmarks[index]);
    if (points.some((point) => !point)) continue;
    const visibility = Math.min(...points.map((point) => point.visibility ?? 0));
    sample[`vis${suffix}`] = visibility;
    sample[`z${suffix}`] = points[1].z;
    if (visibility < minVisibility) continue;
    const [hip, knee, ankle] = points.map((point) => ({ x: point.x * width, y: point.y * height }));
    const angle = angleDegrees(hip, knee, ankle);
    if (angle !== null) sample[`ang${suffix}`] = angle;
  }
  return sample;
}

/** Medijan-filter preko valjanih vrijednosti; NaN ostaju NaN (median_smooth u Pythonu). */
export function medianSmooth(values, k = 5) {
  const out = [...values];
  const indices = values.flatMap((value, index) => (Number.isNaN(value) ? [] : [index]));
  if (indices.length < k) return out;
  const valid = indices.map((index) => values[index]);
  const half = Math.floor(k / 2);
  const padded = [
    ...Array(half).fill(valid[0]),
    ...valid,
    ...Array(half).fill(valid[valid.length - 1]),
  ];
  indices.forEach((index, i) => {
    out[index] = median(padded.slice(i, i + k));
  });
  return out;
}

/** Frameovi u sekundi iz medijana razmaka uzoraka (živa kamera nema stalni fps). */
export function estimateFps(times) {
  if (times.length < 2) return 30;
  const steps = times.slice(1).map((time, index) => time - times[index]).filter((step) => step > 0);
  return steps.length ? 1 / median(steps) : 30;
}

/**
 * Lokalni maksimumi s minimalnim razmakom i prominencijom (find_peaks u Pythonu).
 * Prominencija = vrh minus viši od dva minimuma u prozoru ±winS.
 */
export function findPeaks(values, fps, { minDistS = 0.35, minProm = 10, winS = 0.6 } = {}) {
  const y = values.map((value) => (Number.isFinite(value) ? value : -Infinity));
  const n = y.length;
  const w = Math.max(1, Math.floor(winS * fps));
  const peaks = [];
  for (let i = 1; i < n - 1; i += 1) {
    if (!(y[i] >= y[i - 1] && y[i] > y[i + 1] && Number.isFinite(y[i]))) continue;
    const before = y.slice(Math.max(0, i - w), i).filter(Number.isFinite);
    const after = y.slice(i + 1, i + 1 + w).filter(Number.isFinite);
    if (!before.length || !after.length) continue;
    if (y[i] - Math.max(Math.min(...before), Math.min(...after)) >= minProm) peaks.push(i);
  }
  const minDistance = Math.floor(minDistS * fps);
  const kept = [];
  for (const i of [...peaks].sort((a, b) => y[b] - y[a])) {
    if (kept.every((j) => Math.abs(i - j) >= minDistance)) kept.push(i);
  }
  return kept.sort((a, b) => a - b);
}

/**
 * Vrhovi fleksije koji predstavljaju ponavljanje (čučanj, iskorak) ili zamah u hodu.
 * U hodu koljeno ima mali vrh u osloncu i veliki u zamahu, pa ostaju samo vrhovi iznad pola maksimuma.
 */
export function flexionPeaks(flexion, fps, minProm = 20) {
  const peaks = findPeaks(flexion, fps, { minProm });
  if (!peaks.length) return peaks;
  const threshold = 0.5 * Math.max(...peaks.map((i) => flexion[i]));
  return peaks.filter((i) => flexion[i] >= threshold);
}

/** Kut odabrane noge po frameu: left | right | near (po frameu bliža kameri, manji z). */
export function selectLeg(samples, side) {
  if (side === "left") return samples.map((sample) => sample.angL);
  if (side === "right") return samples.map((sample) => sample.angR);
  return samples.map((sample) => {
    const nearLeft = sample.zL <= sample.zR;
    const near = nearLeft ? sample.angL : sample.angR;
    return Number.isNaN(near) ? (nearLeft ? sample.angR : sample.angL) : near;
  });
}

/** Opseg pokreta odabrane noge, ponavljanja i treperenje (ispis knee_angle.py). */
export function analyzeKnee(samples, side) {
  const raw = selectLeg(samples, side);
  const validFrames = raw.filter((value) => !Number.isNaN(value)).length;
  if (validFrames < 5) return { validFrames, totalFrames: samples.length };
  const times = samples.map((sample) => sample.t);
  const fps = estimateFps(times);
  const smoothed = medianSmooth(raw);
  const finite = smoothed.filter((value) => !Number.isNaN(value));
  const minAngle = Math.min(...finite);
  const maxAngle = Math.max(...finite);
  const flexion = smoothed.map((angle) => 180 - angle);
  const repetitions = flexionPeaks(flexion, fps).map((index) => ({
    t: times[index],
    flexion: flexion[index],
  }));
  const jumps = raw
    .slice(1)
    .map((value, index) => Math.abs(value - raw[index]))
    .filter((value) => !Number.isNaN(value));
  const nearLeftShare =
    side === "near" ? samples.filter((sample) => sample.zL <= sample.zR).length / samples.length : null;
  return {
    validFrames,
    totalFrames: samples.length,
    fps,
    times,
    flexion,
    minFlexion: 180 - maxAngle,
    maxFlexion: 180 - minAngle,
    rom: maxAngle - minAngle,
    jitter: jumps.length ? median(jumps) : NaN,
    repetitions,
    nearLeftShare,
  };
}

/** Kadenca iz vrhova fleksije svake noge (gait_stats u Pythonu). */
export function analyzeGait(samples) {
  const times = samples.map((sample) => sample.t);
  const fps = estimateFps(times);
  const legs = {};
  let totalSteps = 0;
  for (const [name, key] of [["L", "angL"], ["R", "angR"]]) {
    const angles = samples.map((sample) => sample[key]);
    if (angles.every(Number.isNaN)) continue;
    const flexion = medianSmooth(angles).map((angle) => 180 - angle);
    const peaks = flexionPeaks(flexion, fps);
    const peakTimes = peaks.map((index) => times[index]);
    const strides = peakTimes.slice(1).map((time, index) => time - peakTimes[index]);
    legs[name] = {
      peaks,
      times: peakTimes,
      flexion: peaks.map((index) => flexion[index]),
      strides,
      strideS: strides.length ? median(strides) : null,
    };
    totalSteps += peaks.length;
  }
  const legCount = Object.keys(legs).length;
  const duration = times.length > 1 ? times[times.length - 1] - times[0] : 0;
  if (!legCount || duration <= 0) return null;
  // ako je vidljiva samo jedna noga, druga radi isto toliko koraka
  const steps = legCount === 2 ? totalSteps : totalSteps * 2;
  const strides = Object.values(legs).flatMap((leg) => leg.strides);
  // kadenca iz medijana ciklusa (2 koraka po ciklusu) — ne ovisi o rubovima klipa
  const cadence = strides.length ? 120 / median(strides) : (steps / duration) * 60;
  return { legs, duration, steps, cadence };
}

function csvField(value) {
  if (typeof value === "number") return Number.isFinite(value) ? String(Number(value.toFixed(4))) : "";
  const text = String(value ?? "");
  return /[";\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

/** CSV sa ";" (Excel s hrvatskim postavkama), decimalna točka kao u Python CSV-u. */
export function toCsv(header, rows) {
  return [header, ...rows].map((row) => row.map(csvField).join(";")).join("\n");
}

/** Frameovi u istom stupčastom obliku kao CSV iz knee_angle.py (+ izglađeni kut odabrane noge). */
export function samplesCsv(samples, side) {
  const smoothed = medianSmooth(selectLeg(samples, side));
  return toCsv(
    ["t_s", "kut_L", "kut_D", "vis_L", "vis_D", "z_koljeno_L", "z_koljeno_D", "kut_odabrani"],
    samples.map((sample, index) => [
      sample.t,
      sample.angL,
      sample.angR,
      sample.visL,
      sample.visR,
      sample.zL,
      sample.zR,
      smoothed[index],
    ]),
  );
}
