/*
"""
This script runs the server based on the Flask package and hosts the GUIs on the web
Parameters
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

// TODO: get only name:value
function form2Json(){
  var formData= $("#regparameters").serializeArray();
  var data2Send={};
  $.each(formData, function(i, field){
    data2Send[field.name]=field.value;
  });
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