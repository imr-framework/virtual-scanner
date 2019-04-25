//Once registration is done it doesn't allow to keep on registering
function autoFillForm(dict){

  $("#subjectSelect").val(dict['subjecttype']);
  if (dict['subjecttype'] == "ISMRM-NIST") {
    var img_element = "<img src='../static/phantom_pics/" + dict['subjecttype'] + ".jpg'/>"
  }
  else {
    var img_element = "<img src='../static/phantom_pics/" + dict['subjecttype'] + ".png'/>"
  }
  $("#subjectImage-container").html(img_element)
  $('#subjectSelect option:not(:selected)').attr('disabled', true);
  $("#patid").val(dict['patid']);
  $("#name").val(dict['name']);
  $("#age").val(dict['age']);
  $("#dob").val(dict['dob']);
  //By now it will always be other
  var gender = dict['gender'];
  $("#"+gender).prop('checked',true);
  //measuring system
  $("#met").prop('checked',true);
  $("#weight").val(dict['weight']);
  $("#height").val(dict['height']);
  $("#orient").val(dict['orient']);
  $("#anatomy").val(dict['anatomy']);
  $('#orient option:not(:selected)').attr('disabled', true);
  $('#anatomy option:not(:selected)').attr('disabled', true);



  // TODO: show images, gray out inputs and prevent any change
  $("input").prop("disabled",true);
  $("input").css("background-color","#bfbfbf");
  $(".selector").css("background-color","#bfbfbf");

}
