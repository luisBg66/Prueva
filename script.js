// script.js
// ---------------------------------------------------------
// Variables UI y Modelo
const imageUpload = document.getElementById('image-upload');
const previewImg = document.getElementById('preview');
const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediccion');
const claseInput = document.getElementById('clase');
const btnCapturar = document.getElementById('btnCapturar');
const btnEntrenar = document.getElementById('btnEntrenar');
const btnPredecir = document.getElementById('btnPredecir'); 
const muestrasInfo = document.getElementById('muestrasInfo');

const btnExportarMuestras = document.getElementById('btnExportarMuestras');
const btnImportarMuestras = document.getElementById('btnImportarMuestras');
const btnGuardarLocal = document.getElementById('btnGuardarLocal'); 
const btnExportarModelo = document.getElementById('btnExportarModelo'); 

const esText = document.getElementById('es_text');
const purepechaText = document.getElementById('purepecha_text');
const mayaText = document.getElementById('maya_text');
const otomiText = document.getElementById('otomi_text');


// Configuración del Modelo de Transferencia
const MOBILE_NET_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_100_224/feature_vector/3/default/1'; 
const IMAGE_SIZE = 224;

let featureExtractorModel = null; 
let tfModel = null; 
let entrenado = false;
let muestras = []; 
let clasesUnicas = []; 
let selectedFiles = []; 

// --- DICCIONARIO DE TRADUCCIONES ---
const vocabulario = [
    { clase: 'taza', es: 'Taza', purepecha: 'Jantsïkua', maya: 'Luch', otomi: 'Ndami' },
    { clase: 'boligrafo', es: 'Bolígrafo', purepecha: 'Tz´intz´uni', maya: 'Tsíib', otomi: 'Xiúi' },
    { clase: 'silla', es: 'Silla', purepecha: 'Jantzkua', maya: 'K\'áanche\'', otomi: 'Yäni' },
];

function getTranslation(className) {
    return vocabulario.find(item => item.clase === className);
}

// ---------------------------------------------------------
// 1. Manejo de Carga de Archivos Múltiples
function handleFiles(event) {
    selectedFiles = Array.from(event.target.files);
    if (selectedFiles.length > 0) {
        const reader = new FileReader();
        reader.onload = function(){ previewImg.src = reader.result; };
        reader.readAsDataURL(selectedFiles[0]);

        statusEl.textContent = `Archivos cargados: ${selectedFiles.length}. Asigna una clase.`;
        if (featureExtractorModel) { 
            btnCapturar.disabled = false; 
        }
    } else {
        btnCapturar.disabled = true;
    }
}

// ---------------------------------------------------------
// 2. Extracción de Características
function getFrameTensor() {
    if (!previewImg.src || !previewImg.complete) return null; 

    return tf.tidy(() => {
        return tf.browser.fromPixels(previewImg)
            .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0); 
    });
}

function extractFeatures() {
    if (!featureExtractorModel) throw new Error("Extractor de características no cargado.");
    
    const inputTensor = getFrameTensor();
    if (!inputTensor) throw new Error("No hay imagen cargada para extraer.");
    
    return tf.tidy(() => {
        const features = featureExtractorModel.predict(inputTensor);
        return features.dataSync(); 
    });
}

// ---------------------------------------------------------
// 3. Captura Múltiple de Muestras
btnCapturar.addEventListener('click', async () => {
    const clase = (claseInput.value || '').toLowerCase().trim();
    if (!clase) { alert('Introduce el nombre del objeto.'); return; }
    if (selectedFiles.length === 0) { alert('Carga imágenes primero.'); return; }
    
    btnCapturar.disabled = true;
    let newSamplesCount = 0;
    
    for (const file of selectedFiles) {
        statusEl.textContent = `Procesando ${file.name}...`;
        
        await new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = function(e){
                previewImg.src = e.target.result;
                
                previewImg.onload = function() {
                    try {
                        const features = extractFeatures(); 
                        muestras.push({ clase, input: features });
                        newSamplesCount++;
                    } catch (error) { console.error(`Error procesando ${file.name}:`, error); }
                    resolve();
                };
            };
            reader.readAsDataURL(file);
        });
    }

    muestrasInfo.textContent = `Muestras capturadas: ${muestras.length}`;
    statusEl.textContent = `✅ Se agregaron ${newSamplesCount} muestras de "${clase}". Total: ${muestras.length}`;
    
    btnCapturar.disabled = false;
    btnEntrenar.disabled = false;
});


// ---------------------------------------------------------
// 4. Entrenamiento del Modelo (MLP)
btnEntrenar.addEventListener('click', async () => {
  if (muestras.length < 10) { alert('Captura por lo menos 10 muestras (mínimo 2 objetos).'); return; }

  statusEl.textContent = '🧠 Preparando y entrenando...';
  clasesUnicas = Array.from(new Set(muestras.map(m => m.clase)));
  if (clasesUnicas.length < 2) { alert('Necesitas al menos 2 objetos diferentes.'); return; }

  const X = tf.tensor(muestras.map(m => m.input)); 
  const yIdx = muestras.map(m => clasesUnicas.indexOf(m.clase));
  const y = tf.tensor1d(yIdx, 'int32');
  const yOneHot = tf.oneHot(y, clasesUnicas.length); 

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, inputShape: [1024], activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: clasesUnicas.length, activation: 'softmax' })); 

  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await model.fit(X, yOneHot, { epochs: 10, shuffle: true, verbose: 1, batchSize: Math.min(16, muestras.length) });

  tfModel = model;
  entrenado = true;
  statusEl.textContent = '✅ Entrenamiento completado. ¡Modelo listo!';
  btnGuardarLocal.disabled = false;
  btnExportarModelo.disabled = false;
  btnPredecir.disabled = false; 
  
  X.dispose(); y.dispose(); yOneHot.dispose();
});


// ---------------------------------------------------------
// 5. Predicción y Traducción
function predictImage() {
    if (!entrenado || !tfModel) { predEl.textContent = 'Modelo no entrenado'; return; }
    const features = extractFeatures(); 
    if (!features) { statusEl.textContent = 'Carga una imagen para identificar.'; return; }
    
    tf.tidy(() => {
        const t = tf.tensor([features]);
        const out = tfModel.predict(t);
        const maxIdx = out.argMax(1).dataSync()[0];
        const conf = out.dataSync()[maxIdx]; 
        
        const predictedClass = clasesUnicas[maxIdx];
        const translations = getTranslation(predictedClass);

        predEl.textContent = `${predictedClass} (${(conf * 100).toFixed(1)}%)`;
        
        if (translations) {
            esText.textContent = translations.es;
            purepechaText.textContent = translations.purepecha;
            mayaText.textContent = translations.maya;
            otomiText.textContent = translations.otomi;
        } else {
            esText.textContent = `Clase ${predictedClass} no tiene traducción mapeada.`;
            purepechaText.textContent = '--';
            mayaText.textContent = '--';
            otomiText.textContent = '--';
        }
    });
    statusEl.textContent = 'Identificación completada.';
}

// Event Listener para la identificación manual
btnPredecir.addEventListener('click', () => {
    if (!entrenado) { alert("El modelo no ha sido entrenado o cargado."); return; }
    predictImage();
});


// ---------------------------------------------------------
// 6. GESTIÓN DE ARCHIVOS (CORRECCIÓN DE EXPORTACIÓN)
// ---------------------------------------------------------

// A. Exportar Muestras (CORREGIDO para serializar Float32Array)
btnExportarMuestras.addEventListener('click', () => {
    if (muestras.length === 0) { alert('No hay muestras para exportar.'); return; }
    
    // CONVERSIÓN CRÍTICA: Convertir cada Float32Array a un Array estándar para JSON
    const exportableMuestras = muestras.map(muestra => {
        return {
            clase: muestra.clase,
            input: Array.from(muestra.input) // Array.from() fuerza la conversión
        };
    });

    const dataStr = JSON.stringify(exportableMuestras);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'muestras_objetos.json'; 
    a.click();
    
    alert(`Se exportaron ${muestras.length} muestras.`);
});

// B. Importar Muestras (Con verificación y corrección de tipo)
btnImportarMuestras.addEventListener('click', () => {
    const inputFile = document.createElement('input');
    inputFile.type = 'file';
    inputFile.accept = '.json';
    inputFile.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const text = await file.text();
        try {
            const arr = JSON.parse(text);
            
            if (!Array.isArray(arr)) {
                throw new Error('El archivo no contiene un array de muestras.');
            }
            
            // VERIFICACIÓN Y CONVERSIÓN: Asegurar que el input sea un Float32Array de 1024
            const cleanedMuestras = arr.map(muestra => {
                if (!muestra.input || muestra.input.length !== 1024) {
                    return null;
                }
                // Convertir la entrada (que viene como Array estándar) a Float32Array
                muestra.input = new Float32Array(muestra.input);
                return muestra;
            }).filter(m => m !== null); 

            if (cleanedMuestras.length === 0) {
                 throw new Error('No se encontraron muestras válidas (1024 dimensiones) para importar.');
            }
            
            muestras = cleanedMuestras;
            muestrasInfo.textContent = `Muestras capturadas: ${muestras.length}`;
            statusEl.textContent = `Muestras importadas: ${muestras.length}. ¡Listo para entrenar!`;
            btnEntrenar.disabled = false;

        } catch (err) {
            alert('Error importando muestras: ' + err.message);
            console.error(err);
        }
    };
    inputFile.click();
});

// C. Guardar Modelo en LocalStorage
btnGuardarLocal.addEventListener('click', async () => {
  if (!entrenado || !tfModel) { alert('Entrena primero'); return; }
  await tfModel.save('localstorage://clasificador-objetos');
  alert('Modelo guardado en localStorage del navegador.');
});

// D. Exportar Modelo (Archivos Descargables)
btnExportarModelo.addEventListener('click', async () => {
    if (!entrenado || !tfModel) { alert('Entrena primero'); return; }

    statusEl.textContent = 'Exportando modelo...';
    try {
        await tfModel.save('downloads://clasificador-objetos');
        statusEl.textContent = '✅ Modelo exportado y descargado (archivos JSON y BIN).';
    } catch (error) {
        statusEl.textContent = '❌ Error al exportar el modelo.';
        console.error("Error al exportar:", error);
    }
});


// ---------------------------------------------------------
// 7. Inicialización Principal
(async function init() {
  statusEl.textContent = 'Cargando MobileNet Base...';

  featureExtractorModel = await tf.loadGraphModel(MOBILE_NET_URL, { fromTFHub: true });
  featureExtractorModel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  statusEl.textContent = 'MobileNet Base cargado ✅';
  
  // 2. Intentar cargar modelo entrenado previamente
  try {
    const loaded = await tf.loadLayersModel('localstorage://clasificador-objetos');
    tfModel = loaded;
    entrenado = true;
    statusEl.textContent = 'Modelo cargado desde localStorage ✅';
    clasesUnicas = vocabulario.map(v => v.clase); 

    // Habilitar controles si se carga el modelo
    btnGuardarLocal.disabled = false;
    btnExportarModelo.disabled = false; 
    btnPredecir.disabled = false; 

  } catch (e) {
    statusEl.textContent = 'Modelo listo para capturar muestras.';
  }

  // Habilitar la captura si ya hay archivos cargados y el modelo base está listo
  if (selectedFiles.length > 0 && featureExtractorModel) {
    btnCapturar.disabled = false;
  }
  btnEntrenar.disabled = false; 
})();