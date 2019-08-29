/*
"""
This script autopopulates the input fields when phantom is selected as a subject and prevent further modifications
----------

Author: Marina Manso;
Date: 04/12/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

*/

$(document).ready(() => {
  var subjecttype = $("#subjectSelect").val()
  if ($('.phantomopt').is(":selected") == true) {


    $("#patid").val(1+Math.floor(Math.random() * 9999)); //Random numbers
    //Random alpha-numeric user ID
    // FIXME: Not working
    /*var randomId = uuidv1(); // -> v1 UUID
    $("#patid").val(randomId);*/
    $("#patid").prop("readonly", true);


    $("#name").val($("#subjectSelect").val());

    // TODO: change This
    if ($("#subjectSelect").val() == "ISMRM-NIST") {
      var img_element = "<img src='../static/phantom_pics/" + $("#subjectSelect").val() + ".jpg'/>"
    }
    else {
      var img_element = "<img src='../static/phantom_pics/" + $("#subjectSelect").val() + ".png'/>"
    }

    $("#subjectImage-container").html(img_element)

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
    $("#other").prop("checked", true);
    $("#male").prop("disabled", true);
    $("#female").prop("disabled", true);
    $("#weight").val("3");
    $("#weight").prop("readonly", true);
    $("#height").val("20");
    $("#height").prop("readonly", true);
    $("#extraImpFields").hide();
    $('#orient option:not(:selected)').attr('disabled', true);
    $('#anatomy option:not(:selected)').attr('disabled', true);



  }
});




$(document).ready(() => {
  $("#subjectSelect").on("change",function(){

  /*When selecting phantom the input fields are automatically filled and they
  cannot be changed by the user*/
  if ($('.phantomopt').is(":selected") == true) {


    $("#patid").val(1+Math.floor(Math.random() * 9999)); //Random numbers
    //Random alpha-numeric user ID
    // FIXME: Not working
    /*var randomId = uuidv1(); // -> v1 UUID
    $("#patid").val(randomId);*/
    $("#patid").prop("readonly", true);


    $("#name").val($(this).val());

    // TODO: change This
    if ($(this).val() == "ISMRM-NIST") {
      var img_element = "<img src='../static/phantom_pics/" + $(this).val() + ".jpg'/>"
    }
    else {
      var img_element = "<img src='../static/phantom_pics/" + $(this).val() + ".png'/>"
    }

    $("#subjectImage-container").html(img_element)

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
    $("#other").prop("checked", true);
    $("#male").prop("disabled", true);
    $("#female").prop("disabled", true);
    $("#weight").val("3");
    $("#weight").prop("readonly", true);
    $("#height").val("20");
    $("#height").prop("readonly", true);
    $("#extraImpFields").hide();
    $('#orient option:not(:selected)').attr('disabled', true);
    $('#anatomy option:not(:selected)').attr('disabled', true);



  }

});
});
