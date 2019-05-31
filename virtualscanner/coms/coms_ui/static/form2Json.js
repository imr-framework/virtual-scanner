/*
"""
This script creates a JSON string object out of the information retrieved from a form and
send it to the server via Flask when the submission button is pressed by the user
----------
    payload,default = None

Returns
-------
    payload or status on ping from clients

Performs
--------
   tx to client
   rx from client

Unit Test app
-------------
     utest_coms_flask
Author: Marina Manso; Modified by: Sairam Geethanath
Date: 03/27/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

*/

//  TODO:generalize form
$(document).on("click", ".submit-form-btn", function(){

//function form2Json(){
  var formData= $("form").serializeArray();
  var data2Send={};

  $.each(formData, function(i, field){
    data2Send[field.name]=field.value;
  });


  //Alert if any field is empty and prevent submission
  /*
  $.each(data2Send,function(i,field){
   if(data2Send[i]==""){
     //Exception for the oz and in fields when metric system is selected
     if ((data2Send.measuresystem=="metric") && ((i == "weight2")||(i == "height2"))) {
        return;
     }
     else {
       alert("All fields must be filled")
       data2Send = undefined;
     }
   }
  });
  */
  //Alert if any field is empty and prevent submission

// Code to change the dictionary structure in case of metric or Imperial
//It always has to return the values in kg and cm
// TODO: Maybe convert it to a switch when more functionalities are added
if (data2Send["formName"] == "reg"){
  imperial2metric(data2Send);
}





  var jsonData = JSON.stringify(data2Send)


     $.ajax({
        type: 'POST',
        url: '/receiver',
        data: jsonData,
        success: redirectResponse(jsonData),
          //window.location = '/register_success'
          //alert('data: ' + data);

        contentType: "application/json",
        //dataType: 'json',

    });


    event.preventDefault();
	});
	// stop link reloading the page


  //saveText(jsonData,"regdata.json");
//}


//Function to save a JSON file locally
//function saveText(text, filename){
//  var a = document.createElement('a');
//  a.setAttribute('href', 'data:text/plain;charset=utf-u,'+encodeURIComponent(text));
//  a.setAttribute('download', filename);
//  a.click()
//}
function redirectResponse(dataIn_json){
  var form_dict = JSON.parse(dataIn_json)
  console.log(form_dict)
  switch (form_dict['formName']) {
    case 'reg':
      setTimeout(function(){ window.location = '/register' }, 100);

      //$(document).ready(autoFillForm(form_dict));
      break;
    /*case 'acq':

      setTimeout(function(){ window.location = '/acquire_success' }, 100);

      break;*/
    default:

  }





}
