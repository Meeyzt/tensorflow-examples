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

async function plotPredictionLine() {
  const [xs, ys] = tf.tidy(() => {
    const normalizedXs = tf.linspace(0, 1, 100);
    const normalizedYs = model.predict(normalizedXs.reshape([100, 1]));

    const xs = denormalize(normalizedXs, inputMin, inputMax);
    const ys = denormalize(normalizedYs, outputMin, outputMax);

    return [xs.dataSync(), ys.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, index) => ({
    x: val,
    y: ys[index]
  }));

  await plot(houseDatas, "Square feet", predictedPoints);
}

function normalize(tensor, min, max) {
  const featureDims = tensor.shape.length > 1 && tensor.shape[1];

  if(featureDims && featureDims > 1) {
    // Birden fazla ozellik icin

    // tensorlari bol
    const features = tf.split(tensor, featureDims, 1);

    // normalize et ve buyuk kucuk degerlerini bul
    const normalisedTensors = features.map((featureTensor, i)=> {
      normalize(featureTensor,
        min ? min[i]: null,
        max ? max[i]: null
      );

      // return degeri
      const returnTensor = tf.concat(normalisedTensors.map(f => f.tensor), 1);
      const featureMin = normalisedTensors.map(f => f.min);
      const featureMax = normalisedTensors.map(f => f.max);

      return { tensor: returnTensor, min: featureMin, max: featureMax };
    })
  } else {
    // Sadece bir ozelllik icin
    return tf.tidy(() => {
      const min = inputTensor.min();
      const max = inputTensor.min();

      const normalizedTensor = tensor.sub(min).div(max.sub(min));

      return {
        tensor: normalizedTensor,
        min,
        max
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
    inputDim: 1,
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

  if(featureDims && featureDims > 1) {
    // Birden fazla ozellik icin

    // tensorlari bol
    const features = tf.split(tensor, featureDims, 1);

    const denormalized = features.map((featureTensor, i) => denormalize(tensor, min[i], max[i]));

    const returnTensor = tf.concat(denormalized, 1);

    return returnTensor;
  } else {
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
        await plotPredictionLine();
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
    class: item.bedrooms > 2 ? '3+': item.bedrooms
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
  
    const optimizer = tf.train.sgd(0.1); // learning rate
    model.compile({
      loss: 'meanSquaredError',
      optimizer: optimizer,
    });
  
    tfvis.show.modelSummary({ name: 'Model Summary'}, model);
    const layer = model.getLayer(undefined, 0); // ilk layeri almak için
    tfvis.show.layer({ name: "Layer 1" }, layer);

    await plotPredictionLine();

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

      await plotPredictionLine();

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