const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [1, 96, 96, 3], units: 3, activation: 'relu' }));

model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

model.add(tf.layers.dense({ units: 5, activation: 'softmax' }));

module.exports = model;
