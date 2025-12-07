// script.js
// ---------------------------------------------------------
// Variables UI
const imageUpload = document.getElementById('image-upload');
const previewImg = document.getElementById('preview');
const webcamEl = document.getElementById('webcam');
const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediccion');
const claseInput = document.getElementById('clase');
const btnCapturar = document.getElementById('btnCapturar');
const btnEntrenar = document.getElementById('btnEntrenar');
const btnPredecirEstático = document.getElementById('btnPredecirEstático'); 
const btnWebcam = document.getElementById('btnWebcam');
const muestrasInfo = document.getElementById('muestrasInfo');

// Botones de gestión
const btnExportarMuestras = document.getElementById('btnExportarMuestras');
const btnImportarMuestras = document.getElementById('btnImportarMuestras');
const btnGuardarLocal = document.getElementById('btnGuardarLocal'); 
const btnExportarModelo = document.getElementById('btnExportarModelo'); 

// Textos de Traducción
const esText = document.getElementById('es_text');
const purepechaText = document.getElementById('purepecha_text');
const mayaText = document.getElementById('maya_text');
const otomiText = document.getElementById('otomi_text');

// Configuración MobileNet V1
const MOBILE_NET_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_100_224/feature_vector/3/default/1'; 
const IMAGE_SIZE = 224;

// Estado del Sistema
let featureExtractorModel = null; 
let tfModel = null; 
let entrenado = false;
let muestras = []; 
let clasesUnicas = []; 
let selectedFiles = []; 
let isWebcamActive = false;
let webcamStream = null;

// --- DICCIONARIO (40 Objetos) ---
const vocabulario = [
    { clase: 'taza', es: 'Taza', purepecha: 'Jantsïkua', maya: 'Luch', otomi: 'Ndami' },
    { clase: 'boligrafo', es: 'Bolígrafo', purepecha: 'Tz\'intz\'uni', maya: 'Ts\'íib', otomi: 'Xiúi' },
    { clase: 'lapiz', es: 'Lápiz', purepecha: 'Siríratarakua', maya: 'Che\'il ts\'íib', otomi: 'Y\'omfii' },
    { clase: 'libro', es: 'Libro', purepecha: 'Karákata', maya: 'Áanalte\'', otomi: 'Hë\'mi' },
    { clase: 'cuaderno', es: 'Cuaderno', purepecha: 'Siríratarakua', maya: 'Ju\'un', otomi: 'Hë\'mi t\'ofo' },
    { clase: 'tijeras', es: 'Tijeras', purepecha: 'Pikurhukua', maya: 'X-t\'o\'op', otomi: 'Xäxi' },
    { clase: 'botella', es: 'Botella', purepecha: 'Urhu', maya: 'P\'uul', otomi: 'Xalo' },
    { clase: 'cuchara', es: 'Cuchara', purepecha: 'Kutsara', maya: 'X-p\'o\'ch', otomi: 'Kutsara' },
    { clase: 'tenedor', es: 'Tenedor', purepecha: 'Pïreri', maya: 'X-p\'o\'ch che\'', otomi: 'Zá\'i' },
    { clase: 'plato', es: 'Plato', purepecha: 'K\'orunda', maya: 'Lák', otomi: 'Mbo' },
    // Agrega el resto de tus objetos aquí...
];

function getTranslation(className) {
    return vocabulario.find(item => item.clase === className);
}

// ---------------------------------------------------------
// 1. Manejo de Archivos (Estático)
function handleFiles(event) {
    selectedFiles = Array.from(event.target.files);
    if (selectedFiles.length > 0) {
        // Apagar webcam si está prendida para ver la foto
        if(isWebcamActive) stopWebcam();
        
        const reader = new FileReader();
        reader.onload = function(){ 
            previewImg.src = reader.result; 
            previewImg.style.display = 'block';
            webcamEl.style.display = 'none';
        };
        reader.readAsDataURL(selectedFiles[0]);

        statusEl.textContent = `Archivos cargados: ${selectedFiles.length}.`;
        if (featureExtractorModel) btnCapturar.disabled = false; 
    }
}

// ---------------------------------------------------------
// 2. Extracción de Características (Híbrido: IMG o VIDEO)
// sourceElement puede ser previewImg (<img>) o webcamEl (<video>)
function extractFeatures(sourceElement) {
    if (!featureExtractorModel) return null;

    // Verificar si el elemento tiene datos válidos
    if (sourceElement.tagName === 'IMG' && (!sourceElement.src || !sourceElement.complete)) return null;
    if (sourceElement.tagName === 'VIDEO' && sourceElement.readyState < 2) return null;

    return tf.tidy(() => {
        // 1. Convertir píxeles a Tensor
        const inputTensor = tf.browser.fromPixels(sourceElement);
        
        // 2. Redimensionar y normalizar para MobileNet
        const resizedTensor = tf.image.resizeNearestNeighbor(inputTensor, [IMAGE_SIZE, IMAGE_SIZE])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0); 

        // 3. Extraer características (1024 valores)
        const features = featureExtractorModel.predict(resizedTensor);
        return features.dataSync(); 
    });
}

// ---------------------------------------------------------
// 3. Captura Estática (Entrenamiento)
btnCapturar.addEventListener('click', async () => {
    const clase = (claseInput.value || '').toLowerCase().trim();
    if (!clase) { alert('Ponle nombre al objeto.'); return; }
    if (selectedFiles.length === 0) { alert('Sube imágenes primero.'); return; }
    
    btnCapturar.disabled = true;
    let newSamplesCount = 0;
    
    // Procesar cada archivo subido
    for (const file of selectedFiles) {
        statusEl.textContent = `Procesando ${file.name}...`;
        await new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = function(e){
                previewImg.src = e.target.result;
                previewImg.onload = function() {
                    try {
                        const features = extractFeatures(previewImg); 
                        if (features) {
                            muestras.push({ clase, input: features });
                            newSamplesCount++;
                        }
                    } catch (err) { console.error(err); }
                    resolve();
                };
            };
            reader.readAsDataURL(file);
        });
    }

    muestrasInfo.textContent = `Total muestras: ${muestras.length}`;
    statusEl.textContent = `✅ +${newSamplesCount} muestras de "${clase}".`;
    btnCapturar.disabled = false;
    btnEntrenar.disabled = false;
    btnExportarMuestras.disabled = false;
});

// ---------------------------------------------------------
// 4. Entrenamiento (Igual que antes)
btnEntrenar.addEventListener('click', async () => {
  if (muestras.length < 5) { alert('Captura más muestras.'); return; }

  statusEl.textContent = '🧠 Entrenando...';
  
  clasesUnicas = Array.from(new Set(muestras.map(m => m.clase)));
  if (clasesUnicas.length < 2) { alert('Necesitas min. 2 clases diferentes.'); return; }

  const X = tf.tensor(muestras.map(m => m.input)); 
  const yIdx = muestras.map(m => clasesUnicas.indexOf(m.clase));
  const y = tf.tensor1d(yIdx, 'int32');
  const yOneHot = tf.oneHot(y, clasesUnicas.length); 

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, inputShape: [1024], activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: clasesUnicas.length, activation: 'softmax' })); 

  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await model.fit(X, yOneHot, { epochs: 15, shuffle: true, batchSize: 16 });

  tfModel = model;
  entrenado = true;
  statusEl.textContent = '✅ Modelo Entrenado.';
  
  // Habilitar botones de uso
  btnGuardarLocal.disabled = false;
  btnExportarModelo.disabled = false;
  btnPredecirEstático.disabled = false; 
  btnWebcam.textContent = "📹 Activar Webcam (Listo)";
  
  X.dispose(); y.dispose(); yOneHot.dispose();
});

// ---------------------------------------------------------
// 5. Lógica de Predicción
// Función genérica que actualiza la UI con el resultado
function makePrediction(features) {
    if (!entrenado || !tfModel || !features) return;

    tf.tidy(() => {
        const t = tf.tensor([features]);
        const out = tfModel.predict(t);
        const maxIdx = out.argMax(1).dataSync()[0];
        const conf = out.dataSync()[maxIdx]; 
        
        const predictedClass = clasesUnicas[maxIdx];
        const translations = getTranslation(predictedClass);

        // Umbral de confianza
        if (conf > 0.6) {
            predEl.textContent = `${predictedClass} (${(conf * 100).toFixed(0)}%)`;
            if (translations) {
                esText.textContent = translations.es;
                purepechaText.textContent = translations.purepecha;
                mayaText.textContent = translations.maya;
                otomiText.textContent = translations.otomi;
            } else {
                esText.textContent = "Sin traducción";
            }
        } else {
            predEl.textContent = "Incierto...";
        }
    });
}

// A. Predicción Estática (Foto subida)
btnPredecirEstático.addEventListener('click', () => {
    if (!entrenado) { alert("Entrena primero."); return; }
    const features = extractFeatures(previewImg);
    if(features) makePrediction(features);
});

// B. Predicción en Vivo (Webcam Loop)
async function detectLoop() {
    // Si la webcam se apagó, detener
    if (!isWebcamActive) return;

    // Solo predecir si el modelo está listo
    if (entrenado && tfModel && featureExtractorModel) {
        // Extraer características del VIDEO en tiempo real
        const features = extractFeatures(webcamEl);
        if (features) {
            makePrediction(features);
        }
        // Pequeño delay para no saturar la CPU si no hay GPU
        await tf.nextFrame();
    }
    
    requestAnimationFrame(detectLoop);
}

// ---------------------------------------------------------
// 6. Manejo de Webcam
btnWebcam.addEventListener('click', () => {
    if (!isWebcamActive) {
        startWebcam();
    } else {
        stopWebcam();
    }
});

async function startWebcam() {
    if(!featureExtractorModel) return;
    try {
        // Ocultar imagen, mostrar video
        previewImg.style.display = 'none';
        webcamEl.style.display = 'block';
        
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        webcamEl.srcObject = stream;
        webcamStream = stream;
        isWebcamActive = true;
        btnWebcam.textContent = "⏹ Detener Webcam";
        btnWebcam.classList.remove('secondary'); // Hacerlo color primario
        
        statusEl.textContent = "📹 Webcam activa. Detectando...";
        detectLoop(); // Iniciar bucle
    } catch (err) {
        alert("Error de cámara (¿Usas HTTPS/Localhost?): " + err.message);
    }
}

function stopWebcam() {
    isWebcamActive = false;
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
    webcamEl.style.display = 'none';
    previewImg.style.display = 'block'; // Volver a mostrar preview
    btnWebcam.textContent = "📹 Activar Webcam";
    btnWebcam.classList.add('secondary');
    statusEl.textContent = "Webcam detenida.";
}

// ---------------------------------------------------------
// 7. Gestión de Archivos (Importar/Exportar) - Sin cambios lógicos, solo integración
btnExportarMuestras.addEventListener('click', () => {
    if (muestras.length === 0) return;
    const exportable = muestras.map(m => ({ clase: m.clase, input: Array.from(m.input) }));
    const blob = new Blob([JSON.stringify(exportable)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'muestras.json';
    a.click();
});

btnImportarMuestras.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = '.json';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if(!file) return;
        try {
            const arr = JSON.parse(await file.text());
            if(!Array.isArray(arr)) throw new Error("No es array");
            const nuevas = arr.map(m => {
                if(!m.input || m.input.length !== 1024) return null;
                m.input = new Float32Array(m.input);
                return m;
            }).filter(m => m);
            muestras = muestras.concat(nuevas);
            muestrasInfo.textContent = `Total muestras: ${muestras.length}`;
            btnEntrenar.disabled = false;
            btnExportarMuestras.disabled = false;
            alert(`Importadas ${nuevas.length} muestras.`);
        } catch(err) { alert("Error importando: " + err.message); }
    };
    input.click();
});

// Guardar/Exportar Modelo
btnGuardarLocal.addEventListener('click', async () => {
    if(entrenado) await tfModel.save('localstorage://clasificador-objetos');
    alert("Guardado localmente.");
});
btnExportarModelo.addEventListener('click', async () => {
    if(entrenado) await tfModel.save('downloads://clasificador-objetos');
});

// ---------------------------------------------------------
// 8. Inicialización
(async function init() {
  statusEl.textContent = 'Cargando MobileNet...';
  featureExtractorModel = await tf.loadGraphModel(MOBILE_NET_URL, { fromTFHub: true });
  // Calentamiento
  featureExtractorModel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  statusEl.textContent = 'Listo. Sube fotos para entrenar.';

  // Cargar modelo guardado si existe
  try {
    const loaded = await tf.loadLayersModel('localstorage://clasificador-objetos');
    tfModel = loaded;
    entrenado = true;
    // Intentar recuperar clasesUnicas del vocabulario si coinciden
    clasesUnicas = vocabulario.map(v => v.clase); 
    btnGuardarLocal.disabled = false;
    btnExportarModelo.disabled = false;
    btnPredecirEstático.disabled = false;
    statusEl.textContent = 'Modelo previo cargado. Listo para Webcam.';
  } catch (e) { }
})();