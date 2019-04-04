/*
"""
This script autopopulates the input fields when phantom is selected as a subject and prevent further modifications
----------

Author: Marina Manso;
Date: 03/25/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

*/

$(document).ready(() => {
  $("#subjectSelect").on("change",function(){

  /*When selecting phantom the input fields are automatically filled and they
  cannot be changed by the user*/
  if ($('#phantomopt').is(":selected") == true) {


    $("#patid").val(1+Math.floor(Math.random() * 9999)); //Random numbers
    //Random alpha-numeric user ID
    // FIXME: Not working
    /*var randomId = uuidv1(); // -> v1 UUID
    $("#patid").val(randomId);*/
    $("#patid").prop("readonly", true);
    $("#name").val("ISMRM-NIST");
    $("#name").prop("readonly", true);
    $("#age").val("00");
    $("#age").prop("readonly", true);

    //Today's date
    var today = new Date();
    var dd = today.getDate();
    var mm = today.getMonth()+1; //January is 0!
    var yyyy = today.getFullYear();
    /*if(dd < 10){
      dd='0'+dd;
    }
    if(mm < 10){
       mm="0"+mm;
    }*/
    today = mm+'/'+dd+'/'+yyyy;

    $("#dob").val(today);
    $("#dob").prop("readonly", true);
    $("#met").prop("checked", true);
    $("#imp").prop("disabled", true);
    $("#opt3").prop("checked", true);
    $("#opt2").prop("disabled", true);
    $("#opt1").prop("disabled", true);
    $("#weight").val("3");
    $("#weight").prop("readonly", true);
    $("#height").val("20");
    $("#height").prop("readonly", true);
    $("#extraImpFields").hide();
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
