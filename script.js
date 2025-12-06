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
const btnGuardar = document.getElementById('btnGuardar');
const btnPredecir = document.getElementById('btnPredecir'); // Nuevo Botón
const muestrasInfo = document.getElementById('muestrasInfo');

const esText = document.getElementById('es_text');
const purepechaText = document.getElementById('purepecha_text');
const mayaText = document.getElementById('maya_text');
const otomiText = document.getElementById('otomi_text');


// Configuración del Modelo de Transferencia
const MOBILE_NET_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_100_224/feature_vector/3/default/1'; // 1024 características
const IMAGE_SIZE = 224;

let featureExtractorModel = null; // Modelo Base (MobileNet)
let tfModel = null; // Modelo Entrenable (MLP)
let entrenado = false;
let muestras = []; // {clase, input: [1024 valores]}
let clasesUnicas = []; // Nombres de las clases únicas detectadas
let selectedFiles = []; // Archivos seleccionados para la captura

// --- DICCIONARIO DE TRADUCCIONES ---
// EJEMPLO: Reemplaza con tus 30 objetos y traducciones
const vocabulario = [
    { clase: 'taza', es: 'Taza', purepecha: 'Jantsïkua', maya: 'Luch', otomi: 'Ndami' },
    { clase: 'boligrafo', es: 'Bolígrafo', purepecha: 'Tz´intz´uni', maya: 'Tsíib', otomi: 'Xiúi' },
    { clase: 'silla', es: 'Silla', purepecha: 'Jantzkua', maya: 'K\'áanche\'', otomi: 'Yäni' },
    { clase: 'moneda', es: 'Moneda', purepecha: 'K´uini', maya: 'Túumben p\'íit', otomi: 'Hñäki' },
    // ** AÑADE TUS 30 OBJETOS AQUÍ **
];

function getTranslation(className) {
    return vocabulario.find(item => item.clase === className);
}

// ---------------------------------------------------------
// 1. Manejo de Carga de Archivos Múltiples
function handleFiles(event) {
    selectedFiles = Array.from(event.target.files);
    if (selectedFiles.length > 0) {
        // Mostrar la primera imagen en la previsualización
        const reader = new FileReader();
        reader.onload = function(){
            previewImg.src = reader.result;
        };
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
                    } catch (error) {
                        console.error(`Error procesando ${file.name}:`, error);
                    }
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

  await model.fit(X, yOneHot, { 
      epochs: 10, 
      shuffle: true, 
      verbose: 1, // Muestra el progreso del entrenamiento en la consola
      batchSize: Math.min(16, muestras.length) 
  });

  tfModel = model;
  entrenado = true;
  statusEl.textContent = '✅ Entrenamiento completado. ¡Modelo listo!';
  btnGuardar.disabled = false;
  btnPredecir.disabled = false; // HABILITAR EL BOTÓN DE PREDICCIÓN
  
  X.dispose(); y.dispose(); yOneHot.dispose();
});


// ---------------------------------------------------------
// 5. Función de Predicción (Identificación del Objeto)
function predictImage() {
    if (!entrenado || !tfModel) {
        predEl.textContent = 'Modelo no entrenado';
        return;
    }
    
    const features = extractFeatures(); 
    if (!features) {
        statusEl.textContent = 'Carga una imagen para identificar.';
        return;
    }
    
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
            esText.textContent = `Clase ${predictedClass} no mapeada.`;
            purepechaText.textContent = '--';
            mayaText.textContent = '--';
            otomiText.textContent = '--';
        }
    });
    statusEl.textContent = 'Identificación completada.';
}

// Event Listener para la identificación manual
btnPredecir.addEventListener('click', () => {
    if (!entrenado) {
        alert("El modelo no ha sido entrenado o cargado.");
        return;
    }
    predictImage();
});


// ---------------------------------------------------------
// 6. Guardar y Cargar (Persistencia)
btnGuardar.addEventListener('click', async () => {
  if (!entrenado || !tfModel) { alert('Entrena primero'); return; }
  await tfModel.save('localstorage://clasificador-objetos');
  alert('Modelo guardado en localStorage del navegador.');
});


// ---------------------------------------------------------
// 7. Inicialización Principal
(async function init() {
  statusEl.textContent = 'Cargando MobileNet Base...';

  featureExtractorModel = await tf.loadGraphModel(MOBILE_NET_URL, { fromTFHub: true });
  featureExtractorModel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  statusEl.textContent = 'MobileNet Base cargado ✅';
  
  // Cargar modelo guardado
  try {
    const loaded = await tf.loadLayersModel('localstorage://clasificador-objetos');
    tfModel = loaded;
    entrenado = true;
    statusEl.textContent = 'Modelo cargado desde localStorage ✅';
    clasesUnicas = vocabulario.map(v => v.clase); 

    // Habilitar botones de control
    btnGuardar.disabled = false;
    btnPredecir.disabled = false; // Habilitar predicción si se carga el modelo
    btnEntrenar.disabled = false; 

  } catch (e) {
    statusEl.textContent = 'Modelo listo para capturar muestras.';
    btnEntrenar.disabled = false; 
  }

  // Habilitar la captura si ya hay archivos cargados y el modelo base está listo
  if (selectedFiles.length > 0 && featureExtractorModel) {
    btnCapturar.disabled = false;
  }
})();