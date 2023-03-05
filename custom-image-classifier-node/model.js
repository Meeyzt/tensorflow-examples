const tf = require('@tensorflow/tfjs-node');

function createModel(NUM_CLASSES) {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({ inputShape: [80, 60, 3], filters: 16, kernelSize: 60, activation: 'relu' }));

  model.add(tf.layers.maxPooling2d({ poolSize: 20 }));

  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 60, activation: 'relu' }));

  model.add(tf.layers.maxPooling2d({ poolSize: 20 }));

  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 60, activation: 'relu' }));

  model.add(tf.layers.maxPooling2d({ poolSize: 20 }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  return model;
}

module.exports = createModel;
