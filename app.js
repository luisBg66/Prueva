// script.js
const imageUpload = document.getElementById('image-upload');
const previewImg = document.getElementById('preview');
const btnClasificar = document.getElementById('btnClasificar');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');

let model = null;

// ---------------------------------------------------------
// Inicialización: Carga del modelo MobileNet
async function loadMobileNet() {
    statusEl.textContent = 'Cargando modelo MobileNet...';
    try {
        // Carga el modelo pre-entrenado
        model = await mobilenet.load(); 
        statusEl.textContent = 'MobileNet cargado ✅';
        btnClasificar.disabled = false;
    } catch (error) {
        statusEl.textContent = 'ERROR al cargar el modelo.';
        console.error(error);
    }
}

// ---------------------------------------------------------
// Previsualización de la Imagen Cargada
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(){
            previewImg.src = reader.result;
            previewImg.style.display = 'block';
            resultsEl.innerHTML = 'Lista para clasificar.';
        };
        reader.readAsDataURL(file);
    }
}

// ---------------------------------------------------------
// Función de Clasificación
btnClasificar.addEventListener('click', async () => {
    if (!model || previewImg.style.display === 'none') {
        alert('Carga una imagen y espera a que el modelo cargue.');
        return;
    }

    statusEl.textContent = 'Clasificando...';
    btnClasificar.disabled = true;
    resultsEl.innerHTML = '';
    
    try {
        // Usamos model.classify() para clasificar el elemento <img> directamente.
        // Pedimos las 3 predicciones más probables.
        const predictions = await model.classify(previewImg, 3); 

        let outputHTML = '<ul>';
        predictions.forEach(p => {
            // className contiene la etiqueta ImageNet (ej: 'tabby cat', 'coffe mug')
            const probability = (p.probability * 100).toFixed(2);
            outputHTML += `<li><strong>${p.className}</strong>: ${probability}%</li>`;
        });
        outputHTML += '</ul>';

        resultsEl.innerHTML = outputHTML;
        statusEl.textContent = 'Clasificación completada.';

    } catch (error) {
        statusEl.textContent = 'Error durante la clasificación.';
        console.error(error);
    } finally {
        btnClasificar.disabled = false;
    }
});

// ---------------------------------------------------------
// Iniciar la carga
loadMobileNet();