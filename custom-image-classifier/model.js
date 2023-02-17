const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [40000], units: 64, activation: 'relu' }));

model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

export default model;
