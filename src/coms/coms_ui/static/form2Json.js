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
   Tx to client
   Rx from client

Unit Test app
-------------
     utest_coms_flask
Author: Marina Manso; Modified by: Sairam Geethanath
Date: 03/11/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

*/

//  TODO:generalize form
function form2Json(){
  var formData= $("#regparameters").serializeArray();
  var data2Send={};
  $.each(formData, function(i, field){
    data2Send[field.name]=field.value;
  });


  //Alert if any field is empty and prevent submission
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
  //Alert if any field is empty and prevent submission

  // Code to change the dictionary structure in case of metric or Imperial
//It always has to return the values in kg and cm

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


     $.ajax({
        type: 'POST',
        url: '/receiver',
        data: jsonData,
        success: function(data) { alert('data: ' + data); },
        contentType: "application/json",
        dataType: 'json'
    });


    event.preventDefault();
	};
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
