// script.js

// Asegurar que el DOM esté cargado antes de ejecutar el código
window.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const previewImg = document.getElementById('preview');
    const btnClasificar = document.getElementById('btnClasificar');
    const btnCapturar = document.getElementById('btnCapturar');
    const btnEntrenar = document.getElementById('btnEntrenar');
    const btnGuardar = document.getElementById('btnGuardar');
    const muestrasInfo = document.getElementById('muestrasInfo');
    const statusEl = document.getElementById('status');
    const resultsEl = document.getElementById('results');

    let model = null;
    let muestras = []; // almacenará { clase, image }

    // --- DICCIONARIO PERSONALIZADO (mapa de MobileNet -> traducciones) ---
    const clasificacionesPersonalizadas = {
        "ballpoint": { es: 'Bolígrafo', purepecha: 'Tz´intz´uni', maya: 'Tsíib', otomi: 'Xiúi' },
        "ballpoint pen": { es: 'Bolígrafo', purepecha: 'Tz´intz´uni', maya: 'Tsíib', otomi: 'Xiúi' },
        "ballpen": { es: 'Bolígrafo', purepecha: 'Tz´intz´uni', maya: 'Tsíib', otomi: 'Xiúi' },
        "biro": { es: 'Bolígrafo', purepecha: 'Tz´intz´uni', maya: 'Tsíib', otomi: 'Xiúi' },
        "coffee mug": { es: 'Taza', purepecha: 'Jantsïkua', maya: 'Luch', otomi: 'Ndami' },
        "coffee cup": { es: 'Taza', purepecha: 'Jantsïkua', maya: 'Luch', otomi: 'Ndami' },
        // añade más mapeos según tus necesidades
    };

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
                // Habilitar captura cuando la imagen esté lista
                if (btnCapturar) btnCapturar.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }

    // Exponer la función globalmente porque el HTML usa onchange="previewImage(event)"
    window.previewImage = previewImage;

    // ---------------------------------------------------------
    // Capturar muestra (guarda la dataURL y la clase)
    if (btnCapturar) {
        btnCapturar.addEventListener('click', () => {
            const claseInput = document.getElementById('clase');
            const clase = (claseInput.value || '').toLowerCase().trim();
            if (!clase) { alert('Introduce el nombre del objeto.'); return; }
            if (!previewImg.src) { alert('Carga una imagen primero.'); return; }

            muestras.push({ clase, image: previewImg.src });
            if (muestrasInfo) muestrasInfo.textContent = `Muestras capturadas: ${muestras.length}`;
            statusEl.textContent = `Muestra de "${clase}" capturada. Total: ${muestras.length}`;

            // Habilitar botones de entrenamiento/guardar (si más adelante implementas)
            if (btnEntrenar) btnEntrenar.disabled = false;
            if (btnGuardar) btnGuardar.disabled = false;
        });
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
            let foundTranslation = false;

            for (const p of predictions) {
                const probability = (p.probability * 100).toFixed(2);
                const className = p.className.toLowerCase();
                outputHTML += `<li><strong>${p.className}</strong>: ${probability}%</li>`;

                // Buscar coincidencias parciales en el diccionario
                for (const key in clasificacionesPersonalizadas) {
                    if (className.includes(key)) {
                        const t = clasificacionesPersonalizadas[key];
                        resultsEl.innerHTML = `
                            ${outputHTML}
                            <hr>
                            <h2>Traducción Confirmada:</h2>
                            <p><strong>Español:</strong> ${t.es}</p>
                            <p><strong>Purépecha:</strong> ${t.purepecha}</p>
                            <p><strong>Maya:</strong> ${t.maya}</p>
                            <p><strong>Otomí:</strong> ${t.otomi}</p>
                        `;
                        foundTranslation = true;
                        break;
                    }
                }
                if (foundTranslation) break;
            }

            if (!foundTranslation) {
                outputHTML += '</ul>';
                resultsEl.innerHTML = outputHTML + '<hr><p>Traducción: No se encontró un objeto mapeado en el top 3.</p>';
            }

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
});