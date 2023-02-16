const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }));

model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

export default model;
