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
  return tf.tidy(() => {
    const normalizedTensor = tensor.sub(min).div(max.sub(min));

    return {
      tensor: normalizedTensor
    }
  });
}

function createModel() {
  const model = tf.sequential();

  // linear model
  // model.add(tf.layers.dense({
  //   units: 1,
  //   useBias: false,
  //   activation: 'linear',
  //   inputDim: 1,
  // }));

  //sigmoid model
  // model.add(tf.layers.dense({
  //   units: 1,
  //   useBias: true,
  //   activation: 'sigmoid',
  //   inputDim: 1,
  // }));

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
  const denormalizedTensor = tensor.mul(max.sub(min)).add(min)
  return denormalizedTensor;
}

function train(model, inputs, outputs) {
  
  // { onBatchEnd, onEpochEnd }
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
      // onEpochEnd: (epoch, log) => console.log(`Epoch ${ epoch + 1 } => Kayıp: ${ log.loss }`)
      // onBatchEnd: (batch, log) => console.log(`Batch ${ batch + 1 } => Kayıp: ${ log.loss }`) 
    }
  })
}

async function run() {
  // const xs = tf.tensor1d([1, 2, 3]);
  // const ys = xs.mul(tf.scalar(2));

  // ys.print(); //[2, 4, 6]

  // const xs = tf.tensor2d([1, 2, 3, 4, 5, 6], [ 3, 2]);
  // xs.print();
  // /*
  //   [[1, 2],
  //   [3, 4],
  //   [5, 6]]
  //  */

  // Aynı shape olmaları gerekiyor.
  // const ys = xs.mul(tf.tensor2d([7, 8, 9, 10, 11 ,12], [ 3, 2]));

  // ys.print();
  // /*
  //   [[7 , 16],
  //   [27, 40],
  //   [55, 72]]
  // */

  // const xs = tf.tensor2d([172, 232, 123, 255, 233, 232], [ 3, 2]);
  // const ys = xs.div(tf.scalar(255));

  // ys.print();
  /* 
    [[0.6745098, 0.9098039],
    [0.4823529, 1        ],
    [0.9137254, 0.9098039]]
  */

  const houseDataset = tf.data.csv('./kc_house_data.csv');

  let houses = await houseDataset.toArray();

  const tenHouseData = await houseDataset.take(10).toArray();

  if(houses.length %2 !== 0) {
    houses.pop();
  }

  tf.util.shuffle(houses);

  console.log(tenHouseData);

  houseDatas = houses.map((item) => ({
    x: item.sqft_living,
    y: item.price
  }));

  // plot(await houseDatas.toArray());

  const inputValues = await houseDatas.map(h => h.x);
  const inputTensor = tf.tensor2d(inputValues, [inputValues.length, 1]);

  const outputValues = await houseDatas.map(h => h.y);
  const outputTensor = tf.tensor2d(outputValues, [outputValues.length, 1]);

  // inputTensor.print();
  // outputTensor.print()
  inputMin = inputTensor.min();
  inputMax = inputTensor.max();
  outputMin = outputTensor.min();
  outputMax = outputTensor.max();

  const normalizedInput = normalize(inputTensor, inputMin, inputMax);
  const normalizedOutput = normalize(outputTensor, outputMin, outputMax);

  const [normalizedTrainInput, normalizedTestInput] = tf.split(normalizedInput.tensor, 2);
  const [normalizedTrainOutput, normalizedTestOutput] = tf.split(normalizedOutput.tensor, 2);

  // normalizedTrainInput.print(true);

  // normalizedInput.tensor.print();
  // normalizedOutput.tensor.print();

  // denormalize(normalizedInput.tensor, normalizedInput.min, normalizedInput.max).print();

  // const model = createModel();

  // const optimizer = tf.train.sgd(0.1); // learning rate

  // model.compile({
  //   loss: 'meanSquaredError',
  //   optimizer: 'adam',
  // });

  // tfvis.show.modelSummary({ name: 'Model Summary'}, model);

  // const trainResult = await train(model, normalizedTrainInput, normalizedTrainOutput);
  // console.log(trainResult);

  // const trainLoss = trainResult.history.loss.pop(); // arraydeki en son eleman son epoch [-1] de olabilirdi
  // console.log(`Training Loss: ${trainLoss}`);

  // const testLossTensor = model.evaluate(normalizedTestInput, normalizedTestOutput);
  // const testResult = await testLossTensor.dataSync();
  // console.log(`Test Loss: ${testResult}`)
  
  // Modeli görselleştirme
  // tfvis.show.modelSummary({ name: 'Model Summary'}, model);

  // const layer = model.getLayer(undefined, 0); // ilk layeri almak için
  // tfvis.show.layer({ name: "Layer 1" }, layer);

  // model.save(localStoragePath);

  console.log('ready');

  document.getElementById('train').addEventListener('click', async() => {
    model = createModel();
  
    const optimizer = tf.train.sgd(0.1); // learning rate
    // const optimizer = tf.train.adam(0.1);
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