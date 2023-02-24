const data = require('./data');
const model = require('./model');

async function train() {
  await data.loadData();
  const { images, labels } = data.getTestData();

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  let results = await model.fit(images, labels, {
    shuffle: true,
    batchSize: 500,
    epochs: 64,
    callbacks: { onEpochEnd: logProcess }
  });

  data.trainData.dispose();

  evaluate();
}

function logProcess(epoch, logs) {
  console.log('Data for epoch'+ epoch, logs);
}

async function evaluate() {
  const { testImages, testLabels } = data.getTestData();

  let answer = tf.tidy(() => {
    let output = model.predict(testImages[0]);
    output.print();

    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    console.log('result ==>', testLabels[index], testLabels[index]);

    answer.dispose();
  });

  await model.save('./model');

  console.log('evaluate bitti');
}

train();