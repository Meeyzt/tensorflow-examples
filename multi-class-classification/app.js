const localStoragePath = 'localstorage://ev-fiyatlari-multi';
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

async function plotClasses (pointsArray, classKey, size = 400, equalizeClassSizes = false) {
  // Add each class as a series
  const allSeries = {};
  pointsArray.forEach(p => {
    // Add each point to the series for the class it is in
    const seriesName = `${classKey}: ${p.class}`;
    let series = allSeries[seriesName];
    if (!series) {
      series = [];
      allSeries[seriesName] = series;
    }
    series.push(p);
  });
 
  if (equalizeClassSizes) {
    // Find smallest class
    let maxLength = null;
    Object.values(allSeries).forEach(series => {
      if (maxLength === null || series.length < maxLength && series.length >= 100) {
        maxLength = series.length;
      }
    });
    // Limit each class to number of elements of smallest class
    Object.keys(allSeries).forEach(keyName => {
      allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
      if (allSeries[keyName].length < 100) {
        delete allSeries[keyName];
      }
    });
  }
 
  tfvis.render.scatterplot(
    {
      name: `Square feet vs House Price`,
      styles: { width: "100%" }
    },
    {
      values: Object.values(allSeries),
      series: Object.keys(allSeries),
    },
    {
      xLabel: "Square feet",
      yLabel: "Price",
      height: size,
      width: size*1.5,
    }
  );
}

async function plotPredictionHeatMap(name = 'Predicted Class', size = 400) {
  const [valuesPromise, xTicksPromise, yTicksPromise] = tf.tidy(() => {
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
    const normalizedTicksTensor = tf.linspace(0, 1, gridSize);
    const yTicksTensor = denormalize(normalizedTicksTensor, inputMin[0], inputMax[0]);
    const xTicksTensor = denormalize(normalizedTicksTensor.reverse(), inputMin[1], inputMax[1]);

    return [valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array()];
  });

  const values = await valuesPromise;
  const xTicks = await xTicksPromise;
  const yTicks = await yTicksPromise;
  const xTickLabels = xTicks.map((v) => (v / 1000).toFixed(1) + 'k sqft');
  const yTickLabels = yTicks.map((v) => '$' + (v / 1000).toFixed(1)+ 'k');

  tf.unstack(values, 2).forEach((values, i) => {
    const data = {
      values,
      xTickLabels,
      yTickLabels
    };

    tfvis.render.heatmap({
      name: `Bedrooms: ${getClassName(i)} (local)`,
      tab: 'Predictions',
    }, data, {
      height: size
    })

    tfvis.render.heatmap({
      name: `Bedrooms: ${getClassName(i)} (full domain)`,
      tab: 'Predictions',
    }, data, {
      height: size,
      domain: [0, 1]
    })
  });
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
    units: 3,
    useBias: true,
    activation: 'softmax',
  }));

  return model;
}

function denormalize(tensor, min, max) {
  const featureDims = tensor.shape.length > 1 && tensor.shape[1];

  if(featureDims && featureDims > 1) {
    // Birden fazla ozellik icin

    // tensorlari bol
    const features = tf.split(tensor, featureDims, 1);

    const denormalized = features.map((featureTensor, i) => denormalize(featureTensor, min[i], max[i]));

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
        await plotPredictionHeatMap();
        const layer = model.getLayer(undefined, 0); // ilk layeri almak için
        tfvis.show.layer({ name: "Layer 1" }, layer);
      }
    }
  })
}

function getClassIndex(className) {
  if(className === 1 || className === '1') {
    return 0;
  } else if(className === 2 || className === '2') {
    return 1;
  } else {
    return 2;
  }
}

function getClassName(classIndex) {
  if(classIndex === 2) {
    return "3+";
  } else {
    return classIndex + 1;
  }
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

  plotClasses(await houseDatas, 'Bedroom');

  const inputValues = await houseDatas.map(h => [h.x, h.y]);
  const inputTensor = tf.tensor2d(inputValues);

  const outputValues = await houseDatas.map(h => getClassIndex(h.class));
  const outputTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(outputValues, "int32"), 3));

  const normalizedInput = normalize(inputTensor);
  const normalizedOutput = normalize(outputTensor);

  inputMin = normalizedInput.min;
  inputMax = normalizedInput.max;
  outputMin = normalizedOutput.min;
  outputMax = normalizedOutput.max;

  const [normalizedTrainInput, normalizedTestInput] = tf.split(normalizedInput.tensor, 2);
  const [normalizedTrainOutput, normalizedTestOutput] = tf.split(normalizedOutput.tensor, 2);

  normalizedTrainOutput.print();
  console.log(normalizedTrainOutput.shape)

  console.log('ready');

  document.getElementById('train').addEventListener('click', async() => {
    model = createModel();
  
    const optimizer = tf.train.adam(); // learning rate
    model.compile({
      loss: 'categoricalCrossentropy',
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
    const priceInput = Number(document.getElementById('price-input').value);
    const sqftInput = Number(document.getElementById('sqft-input').value);

    if(!priceInput || !sqftInput) {
      alert('Input bos')
      return;
    }

    tf.tidy(() => {
      const predictTensor = tf.tensor2d([[priceInput, sqftInput]]);
      predictTensor.print();
      const normalizedPredictTensor = normalize(predictTensor, inputMin, inputMax);
      const normalizedPredictOutputTensor = model.predict(normalizedPredictTensor.tensor);
      const predictOutputTensor = denormalize(normalizedPredictOutputTensor, outputMin, outputMax);
      const predicts = predictOutputTensor.dataSync();
      let predictString = '';

      
      for (let x = 0; x < 3; x++) {
        predictString  += `Likelihood of having ${getClassName(x)} bedrooms is ${(predicts[x] * 100).toFixed(1)}%<br />`
      }
      console.log(predictString)

      document.getElementById('prediction').innerHTML = `${predictString}`;
    })
  });
}

document.addEventListener('DOMContentLoaded', run);