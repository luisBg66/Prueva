// script.js
// ---------------------------------------------------------
// Variables UI y Modelo
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediccion');
const claseInput = document.getElementById('clase');
const btnCapturar = document.getElementById('btnCapturar');
const btnEntrenar = document.getElementById('btnEntrenar');
const btnGuardar = document.getElementById('btnGuardar');
const btnExportar = document.getElementById('btnExportar');
const btnExportarModelo = document.getElementById('btnExportarModelo');
const btnImportar = document.getElementById('btnImportar');
const muestrasInfo = document.getElementById('muestrasInfo');

const esText = document.getElementById('es_text');
const purepechaText = document.getElementById('purepecha_text');
const mayaText = document.getElementById('maya_text');
const otomiText = document.getElementById('otomi_text');

let featureExtractorModel = null;
let tfModel = null;
let entrenado = false;
let muestras = []; // {clase, input: [1024 valores]}
let clasesUnicas = []; // Nombres de las clases únicas detectadas

// URL del extractor de características de MobileNet V2 (1024 dimensiones de salida)
// Cambiar a MobileNet V1 para un modelo más rápido
const MOBILE_NET_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_100_224/feature_vector/3/default/1';
const IMAGE_SIZE = 224;

// --- DICCIONARIO DE TRADUCCIONES (Mínimo 30 objetos) ---
const vocabulario = [
    // La CLASE debe ser el mismo nombre que el usuario introduce en el input.
    { clase: 'taza', es: 'Taza', purepecha: 'Jantsïkua', maya: 'Luch', otomi: 'Ndami' },
    { clase: 'silla', es: 'Silla', purepecha: 'Jantzkua', maya: 'K\'áanche\'', otomi: 'Yäni' },
    { clase: 'perro', es: 'Perro', purepecha: 'Tzíku', maya: 'Pek\'', otomi: 'K\'uni' },
    { clase: 'casa', es: 'Casa', purepecha: 'K\'umanchikua', maya: 'Naaj', otomi: 'Nggú' },
    // ** AÑADE TUS OTROS 26 OBJETOS AQUÍ (CLASE y TRADUCCIONES) **
];

function getTranslation(className) {
    // Busca la traducción por el nombre de la clase
    return vocabulario.find(item => item.clase === className);
}

// ---------------------------------------------------------
// Camera Setup
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    return true;
  } catch (err) {
    statusEl.textContent = 'Error: Necesitas un contexto seguro (localhost o HTTPS) para la cámara: ' + err.message;
    return false;
  }
}

// ---------------------------------------------------------
// Preprocesamiento de la Imagen y Extracción de Características
function getFrameTensor() {
    // 1. Dibuja el frame actual del video al canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. Convierte el canvas a Tensor y Preprocesa (224x224 y normalizado)
    return tf.tidy(() => {
        return tf.browser.fromPixels(canvas)
            .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0); // Añadir batch dimension [1, 224, 224, 3]
    });
}

function extractFeatures() {
    if (!featureExtractorModel) throw new Error("Extractor de características no cargado.");
    
    // Ejecuta el tensor preprocesado a través de MobileNet para obtener el vector de 1024
    const inputTensor = getFrameTensor();
    
    return tf.tidy(() => {
        const features = featureExtractorModel.predict(inputTensor);
        // Devuelve el vector aplanado de 1024 valores como Array
        return features.dataSync(); 
    });
}

// ---------------------------------------------------------
// Bucle principal: Detección y Predicción
async function detectLoop() {
  // Solo dibuja el video si el modelo extractor aún no se ha cargado
  if (!featureExtractorModel) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    requestAnimationFrame(detectLoop);
    return;
  }

  // Si el modelo secundario ya fue entrenado, intenta predecir
  if (entrenado && tfModel) {
    const features = extractFeatures(); // Vector de 1024
    
    tf.tidy(() => {
        const t = tf.tensor([features]);
        const out = tfModel.predict(t);
        const probs = out.dataSync(); 
        const maxIdx = out.argMax(1).dataSync()[0];
        const conf = probs[maxIdx];
        
        const predictedClass = clasesUnicas[maxIdx];
        const translations = getTranslation(predictedClass);

        predEl.textContent = `${predictedClass} (${(conf * 100).toFixed(1)}%)`;
        
        if (translations) {
            esText.textContent = translations.es;
            purepechaText.textContent = translations.purepecha;
            mayaText.textContent = translations.maya;
            otomiText.textContent = translations.otomi;
        } else {
            esText.textContent = `Clase no mapeada (${predictedClass})`;
        }
    });
    statusEl.textContent = '✅ Clasificando...';
  } else {
    // Si no está entrenado, solo dibuja el video
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    statusEl.textContent = 'En espera de entrenamiento...';
  }

  requestAnimationFrame(detectLoop);
}

// ---------------------------------------------------------
// Manejo UI: Capturar Muestra
btnCapturar.addEventListener('click', () => {
  const clase = (claseInput.value || '').toLowerCase().trim();
  if (!clase) { alert('Introduce el nombre del objeto.'); return; }
  
  try {
    const features = extractFeatures(); // Vector de 1024 valores
    muestras.push({ clase, input: features });
    muestrasInfo.textContent = `Muestras capturadas: ${muestras.length}`;
    statusEl.textContent = `Muestra de "${clase}" capturada. Total: ${muestras.length}`;
  } catch (error) {
    statusEl.textContent = 'Error al capturar. ¿Cámara cargada?';
    console.error(error);
  }
});

// ---------------------------------------------------------
// Entrenar modelo TF.js (MLP)
btnEntrenar.addEventListener('click', async () => {
  if (muestras.length < 15) { alert('Captura por lo menos 15 muestras (¡mínimo 3 por objeto!)'); return; }

  statusEl.textContent = '🧠 Preparando y entrenando...';

  // 1. Obtener clases únicas y asignación de índice
  clasesUnicas = Array.from(new Set(muestras.map(m => m.clase)));
  if (clasesUnicas.length < 2) { alert('Necesitas al menos 2 objetos diferentes.'); return; }

  // 2. Preparar Tensores de Entrada y Salida (One-Hot)
  const X = tf.tensor(muestras.map(m => m.input)); // [N, 1024]
  const yIdx = muestras.map(m => clasesUnicas.indexOf(m.clase));
  const y = tf.tensor1d(yIdx, 'int32');
  const yOneHot = tf.oneHot(y, clasesUnicas.length); // [N, Num_Clases]

  // 3. Crear modelo Secuencial (MLP)
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, inputShape: [1024], activation: 'relu' })); // Simplificado
  model.add(tf.layers.dense({ units: clasesUnicas.length, activation: 'softmax' })); // Output: Num_Clases

  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  // 4. Entrenar
  await model.fit(X, yOneHot, { 
      epochs: 10, // Reducido para optimización
      shuffle: true, 
      verbose: 1, 
      batchSize: Math.min(32, muestras.length) 
  });

  tfModel = model;
  entrenado = true;
  statusEl.textContent = '✅ Entrenamiento completado. ¡A clasificar!';
  alert('Entrenamiento finalizado. El modelo ya puede clasificar.');
  
  // Limpiar tensores de datos para liberar memoria
  X.dispose(); y.dispose(); yOneHot.dispose();
});

// ---------------------------------------------------------
// Guardar Modelo en LocalStorage
btnGuardar.addEventListener('click', async () => {
  if (!entrenado || !tfModel) { alert('Entrena primero'); return; }
  await tfModel.save('localstorage://clasificador-objetos');
  alert('Modelo guardado en localStorage del navegador.');
});

// ---------------------------------------------------------
// Exportar Muestras (Datos)
btnExportar.addEventListener('click', () => {
    if (muestras.length === 0) {
        alert('No hay muestras para exportar.');
        return;
    }
    
    // Convertir el array de muestras a una cadena JSON
    const dataStr = JSON.stringify(muestras);
    
    // Crear un Blob (archivo binario) con el contenido JSON
    const blob = new Blob([dataStr], { type: 'application/json' });
    
    // Crear un enlace temporal para forzar la descarga
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'muestras_objetos.json'; // Nombre del archivo
    
    // Simular el clic para iniciar la descarga
    a.click();
    
    alert(`Se exportaron ${muestras.length} muestras.`);
});

// ---------------------------------------------------------
// Exportar Modelo como Archivos Descargables
btnExportarModelo.addEventListener('click', async () => {
    if (!entrenado || !tfModel) { alert('Entrena primero'); return; }

    statusEl.textContent = 'Exportando modelo...';
    try {
        // Usa 'downloads://' para que el navegador descargue los archivos
        await tfModel.save('downloads://clasificador-objetos');
        statusEl.textContent = '✅ Modelo exportado y descargado.';
    } catch (error) {
        statusEl.textContent = '❌ Error al exportar el modelo.';
        console.error("Error al exportar:", error);
    }
});

// ---------------------------------------------------------
// Importar Muestras (Datos)
btnImportar.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';

    input.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) {
            alert('No se seleccionó ningún archivo.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const importedData = JSON.parse(e.target.result);
                if (!Array.isArray(importedData)) {
                    throw new Error('El archivo no contiene un array válido.');
                }

                muestras = importedData;
                muestrasInfo.textContent = `Muestras importadas: ${muestras.length}`;
                alert(`Se importaron ${muestras.length} muestras correctamente.`);
            } catch (error) {
                alert('Error al importar las muestras. Asegúrate de que el archivo sea válido.');
                console.error(error);
            }
        };

        reader.readAsText(file);
    });

    input.click();
});

// ---------------------------------------------------------
// Inicialización Principal
(async function init() {
  statusEl.textContent = 'Inicializando cámara...';
  const ok = await setupCamera();
  if (!ok) return;

  statusEl.textContent = 'Cargando MobileNet Base...';
  // 1. Cargar el extractor de características
  featureExtractorModel = await tf.loadGraphModel(MOBILE_NET_URL, { fromTFHub: true });
  
  // Ejecutar un tensor cero para inicializar WebGL y evitar retraso en el primer frame
  featureExtractorModel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  // 2. Intentar cargar modelo entrenado previamente
  try {
    const loaded = await tf.loadLayersModel('localstorage://clasificador-objetos');
    tfModel = loaded;
    entrenado = true;
    statusEl.textContent = 'Modelo cargado desde localStorage ✅';
    
    // Si se cargó, necesitamos reconstruir clasesUnicas (un paso extra si guardas/cargas el modelo)
    // Para simplificar, si el modelo está guardado, asume que las clases están en vocabulario
    clasesUnicas = vocabulario.map(v => v.clase); 

  } catch (e) {
    statusEl.textContent = 'Listo para capturar (Modelo base cargado) ✅';
  }

  detectLoop(); // Iniciar el bucle de detección
})();