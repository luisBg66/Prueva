// app.js
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const predictionEl = document.getElementById('prediction');

let model = null;
const IMAGE_SIZE = 224; // Tamaño de entrada estándar para MobileNet

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    // Esperar a que el video cargue completamente
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

async function init() {
  statusEl.textContent = 'Inicializando cámara...';
  if (!await setupCamera()) return;

  statusEl.textContent = 'Cargando modelo MobileNet...';
  // Cargar el modelo MobileNet V1
  model = await mobilenet.load({ version: 1, alpha: 0.5 }); // alpha: 0.5 es más rápido
  statusEl.textContent = 'MobileNet cargado ✅';

  // Iniciar el bucle de detección en video
  detectLoop();
}

async function detectLoop() {
  if (!model) return;

  // 1. Dibujar el video en el canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // 2. Clasificar el frame actual del canvas
  // topk=3 significa que pedimos las 3 mejores predicciones
  const predictions = await model.classify(canvas, 3); 

  // 3. Procesar y mostrar la predicción
  tf.tidy(() => {
    if (predictions && predictions.length > 0) {
      let outputHTML = '';
      
      // Mostrar las 3 mejores predicciones
      predictions.forEach(p => {
        // La clase es lo que MobileNet ya está entrenado a reconocer (ej: 'tabby cat')
        const label = p.className; 
        const probability = (p.probability * 100).toFixed(2);
        
        outputHTML += `
          <div>
            <strong>${label}</strong> (${probability}%)
          </div>
        `;
      });

      predictionEl.innerHTML = outputHTML;
      statusEl.textContent = 'Detectando...';
    } else {
      predictionEl.textContent = 'Esperando objetos...';
    }
  });

  // Llamar a la función de nuevo en el siguiente frame de animación
  requestAnimationFrame(detectLoop);
}

// Iniciar la aplicación
init();