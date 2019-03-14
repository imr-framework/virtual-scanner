
// TODO: get only name:value

function form2Json(){
  var formData= $("#regparameters").serializeArray();
  var data2Send={};
  $.each(formData, function(i, field){
    data2Send[field.name]=field.value;
  });
  var jsonData = JSON.stringify(data2Send)
  //console.log(jsonData)
  //saveText(jsonData,"regdata.json");
}


//Function to save a JSON file locally
/*function saveText(text, filename){
  var a = document.createElement('a');
  a.setAttribute('href', 'data:text/plain;charset=utf-u,'+encodeURIComponent(text));
  a.setAttribute('download', filename);
  a.click()
}*/
