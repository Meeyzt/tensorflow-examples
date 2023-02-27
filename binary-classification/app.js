const localStoragePath = 'localstorage://ev-fiyatlari-tahmin';
let model = null;
let inputMin = null;
let inputMax = null;
let outputMin = null;
let outputMax = null;
let houseDatas = null;

async function plot(trainValues, featureName, predictedValuesArray = null) {
  values = [trainValues.slice(0, 1000)];
  series = ['Orjinal'];

  if(Array.isArray(predictedValuesArray)) {
    values.push(predictedValuesArray);
    series.push('predicted');
  }

  tfvis.render.scatterplot(
    { name: `${featureName} vs House Price` },
    { values, series},
    { 
      xLabel: featureName,
      yLabel: 'Price'
    }
  )
}

async function plotClasses(trainValues, classKey, equalizeClassSizes) {
  const classes = {};

  trainValues.forEach((val) => {
    if(!classes[`${classKey}: ${val.class}`]) {
      classes[`${classKey}: ${val.class}`] = [];
    }

    classes[`${classKey}: ${val.class}`].push({ ...val });
  });

  if(equalizeClassSizes) {
    // Find smallest Class
    let maxLength = null;

    Object.values(classes).forEach((clas) => {
      if(maxLength === null || clas.length < maxLength && clas.length >= 100) {
        maxLength = clas.length;
      }
    });

    // Limit each class to number of elements of smallest class
    Object.keys(classes).forEach(keyName => {
      if(classes[keyName].length < 100) {
        delete classes[keyName];
      }
    });
  }

  tfvis.render.scatterplot(
    { name: 'Square Feet vs House Price' },
    {
      values: Object.values(classes),
      series: Object.keys(classes)
    },
    { 
      xLabel: 'Square Feet',
      yLabel: 'House Price'
    }
  )
}

async function plotPredictionHeatMap(name = 'Predicted Class', size = 400) {
  const valuesPromise = tf.tidy(() => {
    const gridSize = 50;
    const predictionColumns = [];

    for(let colIndex = 0; colIndex < gridSize; colIndex++) {
      // soldan sağa columlar
      const colInputs = [];
      const x = colIndex / gridSize;

      for(let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        // yukarıdan aşağıda satırlar
        const y = (gridSize - rowIndex) / gridSize;
        colInputs.push([x, y]);
      }

      const colPredictions = model.predict(tf.tensor2d(colInputs));
      predictionColumns.push(colPredictions);
    }
    const valuesTensor = tf.stack(predictionColumns);

    return valuesTensor.array();
  });

  const values = await valuesPromise;

  const data = {
    values
  };

  tfvis.render.heatmap({
    name,
    tab: 'Predictions',
  }, data, {
    height: size
  })
}

function normalize(tensor, min = null, max = null) {
  const featureDims = tensor.shape.length > 1 && tensor.shape[1];

  if(featureDims && featureDims > 1) {
    // Birden fazla ozellik icin

    // tensorlari bol
    const features = tf.split(tensor, featureDims, 1);

    // normalize et ve buyuk kucuk degerlerini bul
    const normalizedTensors = features.map((featureTensor, i) =>
      normalize(featureTensor,
        min ? min[i]: null,
        max ? max[i]: null
      ));

    // return degeri
    const returnTensor = tf.concat(normalizedTensors.map(f => f.tensor), 1);
    const featureMin = normalizedTensors.map(f => f.min);
    const featureMax = normalizedTensors.map(f => f.max);

    return { tensor: returnTensor, min: featureMin, max: featureMax };
  } else {
    // Sadece bir ozelllik icin
    return tf.tidy(() => {
      const tensorMin = tensor.min();
      const tensorMax = tensor.max();

      const normalizedTensor = tensor.sub(tensorMin).div(tensorMax.sub(tensorMin));

      return {
        tensor: normalizedTensor,
        min: tensorMin,
        max: tensorMax
      }
    });
  }
}

function createModel() {
  const model = tf.sequential();

  // karmaşık model
  model.add(tf.layers.dense({
    units: 10,
    useBias: true,
    activation: 'sigmoid',
    inputDim: 2,
  }));

  model.add(tf.layers.dense({
    units: 10,
    useBias: true,
    activation: 'sigmoid',
  }));

  model.add(tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'sigmoid',
  }));

  return model;
}

function denormalize(tensor, min, max) {
  const featureDims = tensor.shape.length > 1 && tensor.shape[1];
  console.log(tensor.shape)

  if(featureDims && featureDims > 1) {
    // Birden fazla ozellik icin

    // tensorlari bol
    const features = tf.split(tensor, featureDims, 1);

    const denormalized = features.map((featureTensor, i) => denormalize(featureTensor, min[i], max[i]));

    const returnTensor = tf.concat(denormalized, 1);

    return returnTensor;
  } else {
    console.log(max, min)
    // bir tensor icin
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min)
    return denormalizedTensor;
  }
}

function train(model, inputs, outputs) {
  const { onEpochEnd } = tfvis.show.fitCallbacks(
    { name: 'Training Performance' },
    ['loss']
  )

  return model.fit(inputs, outputs, {
    batchSize: 32,
    epochs: 100,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd,
      onEpochBegin: async() => {
        await plotPredictionHeatMap();
        const layer = model.getLayer(undefined, 0); // ilk layeri almak için
        tfvis.show.layer({ name: "Layer 1" }, layer);
      }
    }
  })
}

async function run() {
  const houseDataset = tf.data.csv('./kc_house_data.csv');

  let houses = await houseDataset.toArray();

  if(houses.length %2 !== 0) {
    houses.pop();
  }

  tf.util.shuffle(houses);

  houseDatas = houses.map((item) => ({
    x: item.sqft_living,
    y: item.price,
    class: item.bedrooms > 2 ? 3: item.bedrooms
  }));

  plotClasses(await houseDatas, 'Bedroom', true);

  const inputValues = await houseDatas.map(h => [h.x, h.y]);
  const inputTensor = tf.tensor2d(inputValues);

  const outputValues = await houseDatas.map(h => h.class);
  const outputTensor = tf.tensor2d(outputValues, [outputValues.length, 1]);

  const normalizedInput = normalize(inputTensor);
  const normalizedOutput = normalize(outputTensor);

  inputMin = normalizedInput.min;
  inputMax = normalizedInput.max;
  outputMin = normalizedOutput.min;
  outputMax = normalizedOutput.max;

  const [normalizedTrainInput, normalizedTestInput] = tf.split(normalizedInput.tensor, 2);
  const [normalizedTrainOutput, normalizedTestOutput] = tf.split(normalizedOutput.tensor, 2);

  console.log('ready');

  document.getElementById('train').addEventListener('click', async() => {
    model = createModel();
  
    const optimizer = tf.train.adam(0.1); // learning rate
    model.compile({
      loss: 'binaryCrossentropy',
      optimizer: optimizer,
    });
  
    tfvis.show.modelSummary({ name: 'Model Summary'}, model);
    const layer = model.getLayer(undefined, 0); // ilk layeri almak için
    tfvis.show.layer({ name: "Layer 1" }, layer);

    await plotPredictionHeatMap();

    const trainResult = await train(model, normalizedTrainInput, normalizedTrainOutput);
    console.log(trainResult);
  
    const trainLoss = trainResult.history.loss.pop(); // arraydeki en son eleman son epoch [-1] de olabilirdi
    console.log(`Training Loss: ${trainLoss}`);
  
    const testLossTensor = model.evaluate(normalizedTestInput, normalizedTestOutput);
    const testResult = await testLossTensor.dataSync();
    console.log(`Test Loss: ${testResult}`)
    
    model.save(localStoragePath);

    console.log('model Trained')
  });
  
  document.getElementById('load').addEventListener('click', async() => {
    const models = await tf.io.listModels();
  
    const modelInfo = models[localStoragePath];

    if(!modelInfo) {
      alert('Model yok')
      return;
    } else {
      model = await tf.loadLayersModel(localStoragePath);
  
      // Modeli görselleştirme
      tfvis.show.modelSummary({ name: 'Model Summary'}, model);
  
      const layer = model.getLayer(undefined, 0); // ilk layeri almak için
      tfvis.show.layer({ name: "Layer 1" }, layer);

      await plotPredictionHeatMap();

      console.log('model Loaded');
    }
  });

  document.getElementById('pred-button').addEventListener('click', async() => {
    const predInput = Number(document.getElementById('pred-input').value);

    if(!predInput) {
      alert('Input bos')
      return;
    }

    tf.tidy(() => {
      const predictTensor = tf.tensor1d([predInput]);
      const normalizedPredictTensor = normalize(predictTensor, inputMin, inputMax);
      const normalizedPredictOutputTensor = model.predict(normalizedPredictTensor.tensor);
      const predictOutputTensor = denormalize(normalizedPredictOutputTensor, outputMin, outputMax);
      const predictedValue = predictOutputTensor.dataSync()[0];
      const roundedPredictValue = (predictedValue / 1000).toFixed(0) * 1000;

      document.getElementById('prediction').innerText = `$${roundedPredictValue}`;
    })
  });
}

document.addEventListener('DOMContentLoaded', run);