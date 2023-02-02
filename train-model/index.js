async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));

  return cleaned;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

/* 
  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
*/

  const model = createModel();
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;
  
  // tfvis.show.modelSummary({ name: 'Model Summary'}, model)
  
  // train model
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
}

function createModel() {
  const model = tf.sequential();
  // input için yaptık
  // dense olması matrisle yani ağırlık ve önyargı ekleyen bir katman türü
  // inputShape 1 yapıyoruz çünkü girdi olarak 1 sayımız var
  // units ağırlığı belirtiyor
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true}));
  // output için yaptık
  model.add(tf.layers.dense({ units: 1, useBias: true}));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
     // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [ inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [ labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = inputTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin
    };
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  });

  /*
    Optimizer önemli, Burda adam optimizer'ı seçme nedenimiz
    konfigürasyon gerektirmemesi ve pratikte çok etkili olmasıdır.

    loss: Model partilerden her birini öğrenirken ne kadar başarılı gösterir
    meanSquaredError kullanma nedenimiz; model tarafından yapılan tahminleri
    gerçek değerlele karşılaştırmak.
  */

    const batchSize = 32;
    const epochs = 50;

    /*
      batchSize: her döngüde göreceği veri alt kümelerinin boyutunu belirler
      epochs: modelin veri kümesine kaç defa bakacağını ifade eder.
    */

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
    /*
      model.fit: eğitim döngüsünü başlatmak için yazdığımız işlevdir.
      promise ile işlem yapar

      callbacks: eğitim ilerlemesini görmek için yollanır.

      tfvis.show.fitCallback: daha önce verdiğimiz mse ve loss
      için grafikler çizen işlevler oluşturmak için kullandık
    */

}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [ xs, preds] = tf.tidy(() => {
    //
    const xsNorm = tf.linspace(0, 1, 100);
    // Modeli besledik afied model
    const predictions = model.predict(xsNorm.reshape([100, 1]));
    // 100, 1 demek kaç tane örnek var, kaç özellik var bir örnekte
    // Un-normalize the data
    const unNormXs =
      xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

      const unNormPreds =
      predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

      // dataSync tensörde saklanan değerlerin tip dizisini elde etmek
      // için kullanabileceğimiz bir yöntemdir. Bu değerleri JS'de işlememize olanak tanır.
      // genellikle tercih edilen .data() yönteminin senkronudur.
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));


  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}

document.addEventListener('DOMContentLoaded', run);