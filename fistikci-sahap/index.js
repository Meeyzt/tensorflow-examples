async function getData() {
  const images = [];
  const labels = [];

  for (let i = 1; i < 30; i += 1) {
    const labelArray = ['Kirmizi', 'Siirt'];

    for(label of labelArray) {
      const name = label.toLowerCase();
      const image = await captureImage(`data/${name}/${name} ${i}.jpg`);
      images.push(image);
      labels.push(label);
    }
  }

  return [images, labels];
}

async function getTestImages() {
  const images = [];
  const labels = [];

  for (let i = 1; i < 7; i += 1) {
    const labelArray = ['Kirmizi', 'Siirt'];

    for(label of labelArray) {
      const name = label.toLowerCase();
      const image = await captureImage(`data/${name}/${name} (${i}).jpg`);
      images.push(image);
      labels.push(label);
    }
  }
  return [images, labels];
}

function captureImage(imgUrl) {
  const image = new Image();
  image.src = imgUrl;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);

  const imageData = ctx.getImageData(0, 0, 60, 60);
  const trainImage = tf.browser.fromPixels(imageData, 3);
  const trainim =  trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

  return trainim;
}

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = await nextTestBatch(3);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([600, 600, 3]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 600;
    canvas.style = 'margin: 4px;';
    canvas.crossOrigin = "Anonymous";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 600;
  const IMAGE_HEIGHT = 600;
  const IMAGE_CHANNELS = 3;  
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 2;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model, data) {
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 10;
  const TEST_DATA_SIZE = 5;
  // Burada hangi metricleri izleyeceğimize karar veriyoruz. DEBUG Yani
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training',
    tab: 'Model',
    styles: { height: '1000px' }
  };
  
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const trainData = await nextTrainBatch(TRAIN_DATA_SIZE);
  const [trainXS, trainYs] = tf.tidy(() => {
    return [
      trainData.xs.reshape([TRAIN_DATA_SIZE, 600, 600, 3]),
      trainData.labels
    ];
  });

  const testData = await nextTestBatch(TEST_DATA_SIZE);
  const [testXs, testYs] = tf.tidy(() => {
    return [
      testData.xs.reshape([TEST_DATA_SIZE, 600, 600, 3]),
      testData.labels
    ];
  });

  return model.fit(trainXS, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

const classNames = ['Kirmizi', 'Siirt'];

async function doPrediction(model, testDataSize = 5) {
  const IMAGE_WIDTH = 600;
  const IMAGE_HEIGHT = 600;
  const testData = await nextTestBatch(testDataSize);
  const testXs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 3]);
  // argMax: bize ne yüksek olasılık değerinin indexini verir.
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testXs).argMax(-1);

  testXs.dispose();
  
  // en yüksek olasılığı tahmin olarak kullanıyoruz
  return {preds, labels};

  /*
    Burada herhangi bir olasılık eşiği kullanmıyoruz.
    Nispeten düşük de olsa en yüksek değeri alıyoruz.
    Bu projenin ilginç bir uzantısı, bazı gerekli minimum olasılıkları ayarlamak
    ve hiçbir sınıf bu sınıflandırma eşiğini karşılamıyorsa 'rakam bulunamadı' ifadesini belirtmek olacaktır.
  */
}

//Bir dizi tahmin ve etiketle, her sınıf için doğruluğu hesaplayabiliriz.
async function showAccuracy(model) {
  const prediction = await doPrediction(model);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(prediction.labels, prediction.preds);
  const container = { name: 'Accuracy', tab: 'Evaluation' };
  
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  
  prediction.labels.dispose();
}

async function showConfusion(model) {
  const prediction = await doPrediction(model);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(prediction.labels, prediction.preds);
  const container = { name: 'Confusion Matrix', tab: 'Evaluation' };

  tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

  prediction.labels.dispose();
}

async function run() {
  const data = await getData();
  const model = getModel();

  //await showExamples(data);

  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);

  const trainedModel = await train(model, data);
  
  // Değerlendirmeyi göster
  await showAccuracy(model, data);
  await showConfusion(model, data);
}

run();

async function nextTrainBatch(batchSize) {
  let shuffledTrainIndex = 0;
  const trainIndices = tf.util.createShuffledIndices(100);
  const [trainImages, trainLabels] = await getData();

  return nextBatch(
    batchSize, [trainImages, trainLabels], () => {
      shuffledTrainIndex =
          (shuffledTrainIndex + 1) % trainIndices.length;
      return trainIndices[shuffledTrainIndex];
    });
}

async function nextTestBatch(batchSize) {
  let shuffledTestIndex = 0;
  const testIndices = tf.util.createShuffledIndices(batchSize);

  const [testImages, testLabels] = await getTestImages();

  return nextBatch(batchSize, [testImages, testLabels], () => {
    shuffledTestIndex =
        (shuffledTestIndex + 1) % 14;
    return testIndices[shuffledTestIndex];
  });
}

function nextBatch(batchSize, data, index) {
  const IMAGE_SIZE = 1080000;
  const NUM_CLASSES = 2;
  const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
  const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

  for (let i = 0; i < batchSize; i++) {
    const idx = index();

    const image =
        data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
    batchImagesArray.set(image, i * IMAGE_SIZE);

    const label =
        data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
    batchLabelsArray.set(label, i * NUM_CLASSES);
  }

  const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
  const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

  return {xs, labels};
}

async function getModifiedMobilenet()
{
  const trainableLayers = ['denseModified', 'conv_pw_13_bn', 'conv_pw_13', 'conv_dw_13_bn', 'conv _dw_13'];

  const mobilenet = await
  tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  console.log('Mobilenet model is loaded')

  const x = mobilenet.getLayer('global_average_pooling2d_1');

  const predictions = tf.layers.dense({units: 2,  activation: 'softmax',name: 'denseModified'}).apply(x.output);

    let mobilenetModified = tf.model({inputs: mobilenet.input, outputs:  predictions, name: 'modelModified' });

    console.log('Mobilenet model is modified')

    mobilenetModified =
    freezeModelLayers(trainableLayers,mobilenetModified)

    console.log('ModifiedMobilenet model layers are freezed')

    mobilenetModified.compile({loss: 'categoricalCrossentropy',  optimizer: tf.train.adam(1e-3), metrics:   ['accuracy','crossentropy']});

    mobilenet.dispose();

    return mobilenetModified;
}

function freezeModelLayers(trainableLayers,mobilenetModified)
{
  for (const layer of mobilenetModified.layers) 
  {
    layer.trainable = false;
    for (const tobeTrained of trainableLayers) 
    {
      if (layer.name.indexOf(tobeTrained) === 0) 
      {
        layer.trainable = true;
        break;
      }
    }
  }
  return mobilenetModified;
}