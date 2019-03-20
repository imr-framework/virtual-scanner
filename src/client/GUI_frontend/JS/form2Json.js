
// Create a JSON object from form data

// TODO: Generalize this function for all forms
function form2Json(){
  var formData= $("#regparameters").serializeArray();
  var data2Send={};
  $.each(formData, function(i, field){
    data2Send[field.name]=field.value;
  });

if (data2Send["measuresystem"]=="metric"){
  delete data2Send.weight2;
  delete data2Send.height2;
  delete data2Send.measuresystem;
}
else if (data2Send["measuresystem"]=="imperial") {
  var weight_t = data2Send.weight*0.453592 + data2Send.weight2*0.0283495231;
  data2Send.weight = weight_t.toFixed(1);
  delete data2Send.weight2;

  var height_t = data2Send.height*30.48 + data2Send.height2*2.54;
  data2Send.height = height_t.toFixed();
  delete data2Send.height2;

  delete data2Send.measuresystem;

}

  var jsonData = JSON.stringify(data2Send)


  console.log(jsonData)
  //saveText(jsonData,"regdata.json");

}


//Function to save a JSON file locally
/*function saveText(text, filename){
  var a = document.createElement('a');
  a.setAttribute('href', 'data:text/plain;charset=utf-u,'+encodeURIComponent(text));
  a.setAttribute('download', filename);
  a.click()
}*/
