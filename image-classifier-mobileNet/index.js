const img = document.getElementById('img');
const uploadButton = document.getElementById('upload-button');
const imgInput = document.getElementById('image-input');
const imgContainer = document.getElementById('image-container');
const closeButton = document.getElementById('close-button');
const uploadContainer = document.getElementById('upload-container');
const loading = document.getElementById('image-loading');
const prediction = document.getElementById('image-prediction');
const version = 2;
const alpha = 0.5;
let model = null;


uploadButton.addEventListener('click', () => imgInput.click());

closeButton.addEventListener('click', () => {
  imgContainer.classList.add('hidden');
  uploadContainer.classList.remove('hidden');
});

imgInput.addEventListener('change', (event) => {
  const file = event.target.files[0];

  const reader = new FileReader();

  reader.onload = (e) => {
    img.src = e.target.result;
  }

  reader.readAsDataURL(file);

  classify();
  uploadContainer.classList.add('hidden');
  imgContainer.classList.remove('hidden');
});

function classify() {
  model.classify(img).then(predictions => {
    prediction.textContent = predictions[0].className;
    prediction.classList.remove('hidden');
    loading.classList.add('hidden');
    closeButton.classList.remove('hidden')
  });
}

function modelLoad() {
  mobilenet.load().then(m => {
    model = m;
  });
}

addEventListener('DOMContentLoaded', modelLoad);