// script.js
// ---------------------------------------------------------
// Variables UI y modelo
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediccion');
const letraInput = document.getElementById('letra');
const btnCapturar = document.getElementById('btnCapturar');
const btnEntrenar = document.getElementById('btnEntrenar');
const btnGuardar = document.getElementById('btnGuardar');
const btnExportar = document.getElementById('btnExportar');
const btnImportar = document.getElementById('btnImportar');
const muestrasInfo = document.getElementById('muestrasInfo');
const confBar = document.getElementById('confidenceBar');

let handposeModel = null;
let tfModel = null;
let entrenado = false;
const clases = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"];
let muestras = []; // {letra, input: [63 valores]}

// conexiones para dibujar esqueleto (Handpose usa 21 puntos, 0 = muñeca)
const fingerConnections = [
  [0,1],[1,2],[2,3],[3,4],    // pulgar
  [0,5],[5,6],[6,7],[7,8],    // indice
  [0,9],[9,10],[10,11],[11,12],// medio
  [0,13],[13,14],[14,15],[15,16],// anular
  [0,17],[17,18],[18,19],[19,20] // meñique
];

// ---------------------------------------------------------
// Camera
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 } });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    return true;
  } catch (err) {
    statusEl.textContent = 'Error accediendo a la cámara: ' + err.message;
    return false;
  }
}

// ---------------------------------------------------------
// Flatten landmarks from [ [x,y,z], ... ] -> [x,y,z,x,y,z,...]
function flattenLandmarks(landmarks) {
  return landmarks.flatMap(p => [p[0], p[1], p[2] ?? 0]);
}

// ---------------------------------------------------------
// Normaliza landmarks: resta la muñeca (0) y divide por la máxima distancia
function normalizeLandmarks(flat) {
  // flat length expected 63 (21*3)
  const wx = flat[0], wy = flat[1], wz = flat[2];
  // restar muñeca
  const rel = [];
  for (let i=0;i<flat.length;i+=3) {
    rel.push(flat[i]-wx, flat[i+1]-wy, flat[i+2]-wz);
  }
  // calcular escala (máxima distancia euclidiana)
  let maxd = 0;
  for (let i=0;i<rel.length;i+=3) {
    const d = Math.hypot(rel[i], rel[i+1], rel[i+2]);
    if (d>maxd) maxd = d;
  }
  if (maxd === 0) maxd = 1;
  // dividir por maxd
  return rel.map(v => v / maxd);
}

// ---------------------------------------------------------
// Dibuja video + puntos + esqueleto
function drawHand(landmarks) {
  // landmarks: array of [x,y,z]
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // líneas
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(88,166,255,0.95)';
  fingerConnections.forEach(([a,b]) => {
    const p1 = landmarks[a], p2 = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
  });

  // puntos
  for (let i=0;i<landmarks.length;i++) {
    const [x,y] = landmarks[i];
    ctx.beginPath();
    ctx.fillStyle = (i===0?'#ffdd57':'#00ff99');
    ctx.arc(x,y,5,0,Math.PI*2);
    ctx.fill();
  }
}

// ---------------------------------------------------------
// Loop principal: detecta con handpose y si hay modelo predict
async function detectLoop() {
  if (!handposeModel) return;
  const predictions = await handposeModel.estimateHands(video, true);
  if (predictions && predictions.length>0) {
    const p = predictions[0];
    const landmarks = p.landmarks; // array 21 x [x,y,z]
    drawHand(landmarks);

    // preparar input normalizado para predicción
    const flat = flattenLandmarks(landmarks);
    const input = normalizeLandmarks(flat);

    if (entrenado && tfModel) {
      tf.tidy(() => {
        const t = tf.tensor([input]);
        const out = tfModel.predict(t);
        const probs = out.dataSync(); // array de 26
        const maxIdx = out.argMax(1).dataSync()[0];
        const conf = probs[maxIdx];
        predEl.textContent = clases[maxIdx];
        confBar.style.width = Math.round(conf*100) + '%';
      });
    } else {
      predEl.textContent = '-';
      confBar.style.width = '0%';
    }
    statusEl.textContent = '✋ Mano detectada';
  } else {
    // dibujar solo video si no hay mano
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(video,0,0,canvas.width,canvas.height);
    predEl.textContent = '-';
    confBar.style.width = '0%';
    statusEl.textContent = '❌ Sin mano detectada';
  }

  requestAnimationFrame(detectLoop);
}

// ---------------------------------------------------------
// Manejo UI: capturar muestra
btnCapturar.addEventListener('click', async () => {
  const letra = (letraInput.value || '').toUpperCase();
  if (!letra || letra.length !== 1 || !/[A-Z]/.test(letra)) {
    alert('Introduce una letra válida (A-Z).');
    return;
  }
  if (!handposeModel) { alert('Modelo Handpose no cargado.'); return; }
  const preds = await handposeModel.estimateHands(video, true);
  if (!preds || preds.length===0) { alert('No se detectó mano'); return; }
  const flat = flattenLandmarks(preds[0].landmarks);
  const norm = normalizeLandmarks(flat);
  muestras.push({ letra, input: norm });
  muestrasInfo.textContent = `Muestras: ${muestras.length}`;
});

// ---------------------------------------------------------
// Entrenar modelo TF.js
btnEntrenar.addEventListener('click', async () => {
  if (muestras.length < 6) { alert('Captura por lo menos 6 muestras (mejor 30+)'); return; }

  statusEl.textContent = '🧠 Entrenando... (puede tardar según muestras)';
  // preparar tensores
  const X = tf.tensor(muestras.map(m=>m.input));
  const yIdx = muestras.map(m => clases.indexOf(m.letra));
  const y = tf.tensor1d(yIdx, 'int32');
  const yOneHot = tf.oneHot(y, clases.length);

  // crear modelo sencillo
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, inputShape: [63], activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: clases.length, activation: 'softmax' }));

  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await model.fit(X, yOneHot, { epochs: 60, shuffle: true, verbose: 1, batchSize: Math.min(32, muestras.length) });

  tfModel = model;
  entrenado = true;
  statusEl.textContent = '✅ Entrenamiento completado';
  alert('Entrenamiento finalizado. Prueba haciendo gestos ahora.');
});

// ---------------------------------------------------------
// Guardar / Exportar / Importar
btnGuardar.addEventListener('click', async () => {
  if (!entrenado || !tfModel) { alert('Entrena primero'); return; }
  await tfModel.save('localstorage://asl-model');
  alert('Modelo guardado en localStorage.');
});

btnExportar.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(muestras)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'muestras_asl.json';
  a.click();
});

btnImportar.addEventListener('click', () => {
  const inputFile = document.createElement('input');
  inputFile.type = 'file';
  inputFile.accept = '.json';
  inputFile.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    try {
      const arr = JSON.parse(text);
      if (!Array.isArray(arr)) throw new Error('Formato inválido');
      muestras = arr;
      muestrasInfo.textContent = `Muestras: ${muestras.length}`;
      alert('Muestras importadas.');
    } catch (err) {
      alert('Error importando muestras: ' + err.message);
    }
  };
  inputFile.click();
});

// ---------------------------------------------------------
// Inicialización
(async function init() {
  statusEl.textContent = 'Inicializando cámara...';
  const ok = await setupCamera();
  if (!ok) return;

  statusEl.textContent = 'Cargando Handpose...';
  handposeModel = await handpose.load();
  statusEl.textContent = 'Handpose cargado ✅';

  // intentar cargar modelo guardado (opcional)
  try {
    const loaded = await tf.loadLayersModel('localstorage://asl-model');
    tfModel = loaded;
    entrenado = true;
    statusEl.textContent = 'Modelo cargado desde localStorage ✅';
  } catch (e) {
    // no hay modelo guardado, continuar sin error
  }

  detectLoop();
})();
