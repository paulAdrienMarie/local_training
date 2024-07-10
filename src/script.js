const status = document.getElementById("status");
const fileUpload = document.getElementById("file-upload");
const imageContainer = document.getElementById("image-container");
const input_size = 224;

// read the uploaded file
fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  // Set up a callback when the file is loaded
  reader.onload = function (e2) {
    imageContainer.innerHTML = "";
    const image = document.createElement("img");
    image.src = e2.target.result;
    image.id = "image-id";
    imageContainer.appendChild(image);
    image.onload = function () {
      detect(image);
    };
  };
  reader.readAsDataURL(file);
});

// make a request to the python server to generate a caption of the given image
function detect(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input_image: image.src }),
  })
    .then((response) => response.json())
    .then((data) => {
      displayOutput(data);
    })
    .catch((error) => {
      document.getElementById("result").textContent = "Error: " + error;
    });
}

function displayOutput(data) {
  console.log(data);
  const labels = data.object_labels;

  const generated_text = document.createElement("div");
  generated_text.id = "textarea-id";

  // Create a container for the labels
  const labelsContainer = document.createElement("div");
  labelsContainer.id = "labels-container";

  // Iterate over the dictionary and create paragraphs
  for (const label in labels) {
    if (labels.hasOwnProperty(label)) {
      const p = document.createElement("p");
      p.innerHTML = `${label}: <span class="bold">${labels[label]}</span>`;
      labelsContainer.appendChild(p);
    }
  }

  // Append the labels container to the generated text div
  generated_text.appendChild(labelsContainer);

  // Append the generated text to the labels container
  imageContainer.append(generated_text);

  // Call the displayButtons function (assuming it exists)
  displayButtons();
}

function displayButtons() {
  const textarea = document.getElementById("textarea-id");
  let buttons = document.createElement("div");
  buttons.id = "buttons-id";
  textarea.appendChild(buttons);
  let button_validate = document.createElement("button");
  button_validate.className = "check_button";
  button_validate.id = "button-validate-id";
  button_validate.innerText = "Validate";
  let button_retrain = document.createElement("button");
  button_retrain.className = "check_button";
  button_retrain.id = "button-retrain-id";
  button_retrain.innerText = "Retrain";
  buttons.appendChild(button_validate);
  buttons.appendChild(button_retrain);
  add_Event("button-retrain-id");
}

// Function to add event listener to buttons
function add_Event(id) {

  document.getElementById(id).addEventListener("click", function () {

    if (!document.getElementById("new-output-id")) {
      // element div for training
      let training_div = document.createElement("div");
      training_div.id = "training-div";
      const text_area = document.getElementById("textarea-id");
      text_area.appendChild(training_div);


      // element input for new class proposal 
      let new_output_ = document.createElement("input");
      new_output_.id = "new-output-id";

      // element button to launch request to training endpoint
      let submit_button = document.createElement("button");
      submit_button.type = "submit";
      submit_button.id = "launch-training-button";
      submit_button.innerText = "Launch training";

      training_div.appendChild(new_output_);
      training_div.appendChild(submit_button);

      document.getElementById("launch-training-button").addEventListener("click", function() {
        let new_class = document.getElementById('new-output-id').value;
        launch_training_request(new_class);
      })
    }
  })
}

function launch_training_request(new_class){

  let image = document.getElementById("image-id");
  
  document.getElementById("result").textContent = "Training started with success";

    fetch("/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ "input_image": image.src, "correct_class": new_class }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        document.getElementById("result").textContent = data.message;
      })
      .catch((error) => {
        document.getElementById("result").textContent = "Error: " + error;
      });
}
    