let model;
let signature;
const video = document.getElementById('webcam');
const modelOutput = document.getElementById('modelOutput');

async function loadSignature() {
    try {
        const response = await fetch('https://guardeec.github.io/lobe_ml_demo/model/signature.json');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching the signature file', error);
    }
}


async function loadModel() {
    try {
        await tf.ready(); // Ensures TensorFlow.js is ready
        signature = await loadSignature();
        console.log('Signature:', signature);
        model = await tf.loadLayersModel('https://guardeec.github.io/lobe_ml_demo/model/model.json'); // Updated this line
        console.log('Model loaded:', model);
    } catch (error) {
        console.error('Error loading the model or signature file', error);
        throw error; // Re-throw the error to be caught by the catch in init()
    }
}



let currentStream;

async function setupCamera(facingMode = 'user') {
    if (currentStream) {
        currentStream.getTracks().forEach(track => {
            track.stop();
        });
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode }
    });
    video.srcObject = stream;
    currentStream = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

document.getElementById('toggleCamera').addEventListener('click', () => {
    const facingMode = video.srcObject.getVideoTracks()[0].getSettings().facingMode;
    const newFacingMode = facingMode === 'user' ? 'environment' : 'user';
    setupCamera(newFacingMode);
});


function preprocessImage(imageTensor) {
    if (!signature || !signature.inputs || !signature.inputs.Image_1) {
        throw new Error('Signature file does not have the expected structure.');
    }
    // Get the input shape from signature file
    const [imgHeight, imgWidth] = signature.inputs.Image_1.shape.slice(1, 3);
    // convert image to from 0 to 255 -> -1 to 1
    const normalizedImage = tf.div(imageTensor, tf.scalar(127.5));
    const shiftedImage = tf.sub(normalizedImage, tf.scalar(1));
    // make into a batch of 1 so it is shaped [1, height, width, 3]
    const batchedImage = shiftedImage.expandDims(0);
    // resize to match model input
    const resizedImage = tf.image.resizeBilinear(batchedImage, [imgHeight, imgWidth]);
    return resizedImage;
}

async function predict() {
    return tf.tidy(() => { // Wrap your tensor operations in tf.tidy for automatic cleanup
        const capturedTensor = tf.browser.fromPixels(video);
        const preprocessedImage = preprocessImage(capturedTensor);
        const prediction = model.predict(preprocessedImage);

        // Assuming the model outputs a tensor with shape [1, num_classes]
        const confidences = prediction.dataSync(); // Use dataSync to avoid async/await inside tf.tidy
        const highestIndex = confidences.indexOf(Math.max(...confidences));

        // Correctly access the labels from the signature file
        const label = signature.classes.Label[highestIndex];
        const confidence = confidences[highestIndex];

        // Return the prediction results
        return { label, confidence };
    }); // Tensors created inside this block will be disposed of at the end of the block
}

async function runModel() {
    while (true) {
        const { label, confidence } = await predict();
        modelOutput.innerText = `Label: ${label}, Confidence: ${confidence.toFixed(3)}`;
        await tf.nextFrame(); // This gives the browser a chance to handle other tasks, including garbage collection
    }
}


async function init() {
    await loadModel();
    await setupCamera();
    video.play();
    runModel();
}

init().catch(error => {
    console.error('Failed to initialize', error);
});

