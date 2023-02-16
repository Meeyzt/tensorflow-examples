const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [10, 1000], units: 2, activation: 'relu' }));

model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

export default model;
