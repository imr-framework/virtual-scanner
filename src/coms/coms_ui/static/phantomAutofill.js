/*
"""
This script autopopulates the input fields when phantom is selected as a subject and prevent further modifications
----------

Author: Marina Manso;
Date: 03/18/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

*/

$(document).ready(() => {
  $("#subjectSelect").on("change",function(){

  /*When selecting phantom the input fields are automatically filled and they
  cannot be changed by the user*/
  if ($('#phantomopt').is(":selected") == true) {

    //Changing input fields
    // TODO: Change this to input numbers from 0000 to 9999
    $("#patid").val(1+Math.floor(Math.random() * 9999));
    $("#patid").prop("readonly", true);
    $("#name").val("ISMRM-NIST");
    $("#name").prop("readonly", true);
    $("#age").val("20");
    $("#age").prop("readonly", true);
    $("#dob").val("12/12/2012");
    $("#dob").prop("readonly", true);
    $("#met").prop("checked", true);
    $("#imp").prop("disabled", true);
    $("#opt3").prop("checked", true);
    $("#opt2").prop("disabled", true);
    $("#opt1").prop("disabled", true);
    $("#height").val("20");
    $("#height").prop("readonly", true);
    $('#orient option:not(:selected)').attr('disabled', true);
    $('#anatomy option:not(:selected)').attr('disabled', true);


    //Show a picture of the selected phantom
    var img = document.createElement("img");

    img.src = "../static/images/ISMRM_NIST.jpg";
    var src = document.getElementById("subjectImage");

    src.appendChild(img);
    //$("#subjectImage").html()
  }

});
});
