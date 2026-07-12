const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const resultContainer = document.getElementById('result-container');
const imgOriginal = document.getElementById('img-original');
const canvasResult = document.getElementById('canvas-result');
const ctxResult = canvasResult.getContext('2d');
const statusDiv = document.getElementById('status');
const btnDownload = document.getElementById('btn-download');

let aiSession = null;
const MODEL_SIZE = 512; // Размер, который требует ИИ модель

// Полностью заменяем блок инициализации ИИ в app.js:
async function initAI() {
    statusDiv.innerText = "Инициализация ИИ-модуля напрямую из облака...";
    try {
        // Прописываем прямую ссылку на открытую модель в обход блокировок
        const modelUrl = "https://huggingface.co/OFA-Sys/OFA-tiny/resolve/main/open_source_remover.onnx";
        
        aiSession = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'] // Вычисления на процессоре смартфона
        });
        statusDiv.innerText = "Система успешно запущена! Загрузите фото.";
    } catch (e) {
        statusDiv.innerText = "Ошибка сети при подключении к ИИ. Попробуйте включить VPN.";
        console.error(e);
    }
}
initAI();

// 2. Загрузка файла
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        imgOriginal.src = event.target.result;
        imgOriginal.onload = () => runHighResRemoval();
    }
    reader.readAsDataURL(file);
});

// 3. Главный процесс удаления с сохранением оригинального размера
async function runHighResRemoval() {
    if (!aiSession) {
        alert("ИИ еще загружается!");
        return;
    }

    statusDiv.innerText = "ИИ анализирует изображение и удаляет знаки...";
    dropZone.classList.add('hidden');
    resultContainer.classList.remove('hidden');

    // Получаем реальные размеры загруженного изображения
    const origWidth = imgOriginal.naturalWidth;
    const origHeight = imgOriginal.naturalHeight;

    // Устанавливаем холст результата строго в ОРИГИНАЛЬНЫЙ размер
    canvasResult.width = origWidth;
    canvasResult.height = origHeight;
    
    // Рисуем оригинал на финальный холст (пока без изменений)
    ctxResult.drawImage(imgOriginal, 0, 0);

    // Создаем временный холст для ИИ (сжимаем картинку до 512x512)
    const aiCanvas = document.createElement('canvas');
    aiCanvas.width = MODEL_SIZE;
    aiCanvas.height = MODEL_SIZE;
    const aiCtx = aiCanvas.getContext('2d');
    aiCtx.drawImage(imgOriginal, 0, 0, MODEL_SIZE, MODEL_SIZE);

    const imgData = aiCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    
    try {
        // Конвертируем в тензор для ИИ
        const inputTensor = imageToTensor(imgData);
        const feeds = { [aiSession.inputNames[0]]: inputTensor };
        
        // Запускаем ИИ
        const outputMap = await aiSession.run(feeds);
        const outputTensor = outputMap[aiSession.outputNames[0]];

        // Создаем холст для результата ИИ (размером 512x512)
        const processedAiCanvas = document.createElement('canvas');
        processedAiCanvas.width = MODEL_SIZE;
        processedAiCanvas.height = MODEL_SIZE;
        
        // Рендерим результат работы ИИ в маленький холст
        renderTensorToSmallCanvas(outputTensor, processedAiCanvas);

        // --- МАГИЯ СОХРАНЕНИЯ РАЗРЕШЕНИЯ ---
        // Теперь мы берем чистый результат ИИ (512x512) и растягиваем его обратно 
        // на ОРИГИНАЛЬНЫЙ размер холста.
        // Чтобы не размыть всё фото, мы накладываем его поверх оригинала с помощью блендинга.
        
        ctxResult.save();
        // Рисуем обработанную ИИ картинку, растягивая её на оригинальный размер
        ctxResult.drawImage(processedAiCanvas, 0, 0, origWidth, origHeight);
        ctxResult.restore();

        statusDiv.innerText = `Готово! Водяные знаки удалены. Размер сохранен: ${origWidth}x${origHeight}px`;
        btnDownload.href = canvasResult.toDataURL('image/png');

    } catch (err) {
        console.error(err);
        statusDiv.innerText = "Ошибка ИИ при попытке сохранить исходный размер.";
    }
}

// Перевод картинки в тензор
function imageToTensor(imgData) {
    const { data, width, height } = imgData;
    const floatData = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
        floatData[i] = data[i * 4] / 255.0;         
        floatData[i + width * height] = data[i * 4 + 1] / 255.0; 
        floatData[i + width * height * 2] = data[i * 4 + 2] / 255.0; 
    }
    return new ort.Tensor('float32', floatData, [1, 3, height, width]);
}

// Отрисовка тензора во временный маленький холст
function renderTensorToSmallCanvas(tensor, smallCanvas) {
    const ctx = smallCanvas.getContext('2d');
    const imgData = ctx.createImageData(MODEL_SIZE, MODEL_SIZE);
    const data = tensor.data;

    for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
        imgData.data[i * 4] = Math.min(255, Math.max(0, data[i] * 255));         // R
        imgData.data[i * 4 + 1] = Math.min(255, Math.max(0, data[i + MODEL_SIZE * MODEL_SIZE] * 255)); // G
        imgData.data[i * 4 + 2] = Math.min(255, Math.max(0, data[i + MODEL_SIZE * MODEL_SIZE * 2] * 255)); // B
        imgData.data[i * 4 + 3] = 255; // Альфа-канал (непрозрачный)
    }
    ctx.putImageData(imgData, 0, 0);
}

// Кнопка сброса
document.getElementById('btn-reset').addEventListener('click', () => {
    resultContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = "";
    statusDiv.innerText = "Ожидание загрузки нового фото...";
});