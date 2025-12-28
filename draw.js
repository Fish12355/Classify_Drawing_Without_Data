const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.lineWidth = 4;
ctx.lineCap = "round";
ctx.strokeStyle = "black";
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
let coords = [];

let undoStack = [];
let redoStack = [];

const pieCanvas = document.getElementById("pieChart");
const pieCtx = pieCanvas.getContext("2d");
const legend = document.getElementById("legend");

let model = null;

const classes = [
    "airplane","alarm clock","angel","ant","anvil","apple","arm","axe","banana","basketball",
    "bat","bathtub","beach","bear","bed","bicycle","bird","birthday cake","book","bowtie",
    "bread","broom","bus","butterfly","cactus","car","carrot","castle","cat","circle",
    "clock","cloud","cookie","crab","crown","diamond","dog","donut","door","dragon",
    "drums","duck","ear","eye","fish","flower","fork","grapes","grass","guitar",
    "hamburger","hat","headphones","hot dog","house","ice cream","key","knife","ladder","lantern","light bulb",
    "lollipop","mermaid","monkey","moon","mouse","mug","mushroom","nose","ocean","octopus",
    "owl","paintbrush","palm tree","panda","pear","pizza","rabbit","rain","rainbow","sandwich",
    "saw","scissors","sea turtle","sheep","shovel","skateboard","skull","skyscraper","snowman","spider",
    "square","star","strawberry","submarine","sun","tree","triangle","umbrella","whale"
];

canvas.addEventListener("mousedown", e => startDraw(e));
canvas.addEventListener("mousemove", e => draw(e));
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseout", endDraw);

canvas.addEventListener("touchstart", e => startDraw(e.touches[0]));
canvas.addEventListener("touchmove", e => draw(e.touches[0]));
canvas.addEventListener("touchend", endDraw);

function startDraw(e) {
    saveState();
    drawing = true;
    recordCoord(e);
}

function draw(e) {
    if (!drawing) return;

    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);

    recordCoord(e);
}

function endDraw() {
    drawing = false;
    ctx.beginPath();
    throttledPredict();
}

function recordCoord(e) {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    coords.push({ x, y });
}

function getBoundingBox() {
    if (coords.length === 0) return null;
    const xs = coords.map(p => p.x);
    const ys = coords.map(p => p.y);

    return {
        minX: Math.max(Math.min(...xs) - 20, 0),
        minY: Math.max(Math.min(...ys) - 20, 0),
        maxX: Math.min(Math.max(...xs) + 20, canvas.width),
        maxY: Math.min(Math.max(...ys) + 20, canvas.height)
    };
}

function preprocess(imgData) {
    return tf.tidy(() => {
        let t = tf.browser.fromPixels(imgData, 1).toFloat();

        t = tf.sub(255, t);

        t = tf.image.resizeBilinear(t, [28, 28]);
        t = t.div(255);
        t = tf.clipByValue(t, 0, 1);
        return t.reshape([1, 28, 28, 1]);
    });
}

function getImageTensor() {
    const box = getBoundingBox();
    if (!box) return null;

    const w = box.maxX - box.minX;
    const h = box.maxY - box.minY;

    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const cctx = c.getContext("2d");

    cctx.fillStyle = "black";
    cctx.fillRect(0, 0, w, h);

    cctx.drawImage(canvas, box.minX, box.minY, w, h, 0, 0, w, h);
    const data = cctx.getImageData(0, 0, w, h);

    return preprocess(data);
}

function drawPieChart(topClasses) {
    pieCtx.clearRect(0, 0, pieCanvas.width, pieCanvas.height);
    legend.innerHTML = "";

    if (!topClasses || topClasses.length === 0) {
        pieCanvas.style.visibility = "hidden";
        legend.style.visibility = "hidden";
        return;
    }

    pieCanvas.style.visibility = "visible";
    legend.style.visibility = "visible";

    let total = topClasses.reduce((a, b) => a + b.p, 0);
    let angleStart = 0;

    const colors = ["#ff5959", "#ffad33", "#33cc33", "#3399ff", "#cc33ff"];

    topClasses.forEach((item, i) => {
        const slice = (item.p / total) * Math.PI * 2;
        const angleEnd = angleStart + slice;

        // Draw slice
        pieCtx.beginPath();
        pieCtx.moveTo(110, 110);
        pieCtx.fillStyle = colors[i];
        pieCtx.arc(110, 110, 100, angleStart, angleEnd);
        pieCtx.fill();

        angleStart = angleEnd;

        const row = document.createElement("div");
        row.innerHTML = `
            <span class="colorBox" style="background:${colors[i]}"></span>
            <b>${classes[item.i]}</b> — ${(item.p * 100).toFixed(1)}%
        `;
        legend.appendChild(row);
    });
}

async function loadModel() {
    model = await tf.loadLayersModel("classify_drawing_js/model.json");
}

loadModel();

async function predictDrawing() {
    if (!model) return;

    if (isCanvasEmpty()) {
        pieCanvas.style.visibility = "hidden";
        legend.style.visibility = "hidden";
        return;
    }

    const tensor = getImageTensor();
    if (!tensor) return;

    const prediction = model.predict(tensor);
    const data = prediction.dataSync();

    let top = [...data]
        .map((p, i) => ({ p, i }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5);

    drawPieChart(top);
}

function debounce(fn, delay) {
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), delay);
    };
}

const throttledPredict = debounce(predictDrawing, 140);

function saveState() {
    undoStack.push(canvas.toDataURL());
    redoStack = [];
}

function restoreState(dataUrl) {
    let img = new Image();
    img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        throttledPredict();
    };
    img.src = dataUrl;
}

document.getElementById("undoBtn").onclick = () => {
    if (!undoStack.length) return;

    const last = undoStack.pop();
    redoStack.push(canvas.toDataURL());
    restoreState(last);
};

document.getElementById("redoBtn").onclick = () => {
    if (!redoStack.length) return;

    const next = redoStack.pop();
    undoStack.push(canvas.toDataURL());
    restoreState(next);
};

document.getElementById("clearBtn").onclick = () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    coords = [];
    undoStack = [];
    redoStack = [];

    pieCanvas.style.visibility = "hidden";
    legend.style.visibility = "hidden";
    legend.innerHTML = "";
};

function isCanvasEmpty() {
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < pixels.length; i += 4) {
        if (
            pixels[i] !== 255 ||
            pixels[i + 1] !== 255 ||
            pixels[i + 2] !== 255
        ) return false;
    }
    return true;
}
