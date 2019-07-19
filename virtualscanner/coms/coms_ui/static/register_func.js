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

  $('.gender option:not(:checked)').attr('disabled', true);
  //measuring system
  $("#met").prop('checked',true);
  $("#weight").val(dict['weight']);
  $("#height").val(dict['height']);
  $("#orient").val(dict['orient']);
  $("#anatomy").val(dict['anatomy']);

  $('#orient option:not(:selected)').attr('disabled', true);
  $('#anatomy option:not(:selected)').attr('disabled', true);



  // TODO: show images, gray out inputs and prevent any change
  $("input:not(#log-out,#new-subject,#formId2,.feedback-btn)").prop("disabled",true);

  $("input:not(#log-out,#new-subject,.feedback-btn)").css("background-color","#bfbfbf");
  $(".selector").css("background-color","#bfbfbf");

}

$(document).on("click",".feedback-btn",function(){
  window.open('https://docs.google.com/forms/d/1qXr0tWLCkUgDS_ttTvB_86wP0RO4YntILqlM3_kMl9k/edit?ts=5cf6c494', '_blank');
});

$(document).on("change","#weight-unit",function(){
  var selected_unit = $(this).val();
  if (selected_unit == "lbs") {
    if( !$("#weight").val() ) {
    }
    else{
      var kg = $("#weight").val();
      var lbs = kg * 2.205;
      lbs.toFixed(1);
      $("#weight").val(lbs);
    }
  }
  else{
    if( !$("#weight").val() ) {
    }
    else{
      var lbs = $("#weight").val();
      var kg = lbs / 2.205;
      kg.toFixed(1);
      $("#weight").val(kg);
    }
  }
});

$(document).on("change","#height-unit",function(){
  var selected_unit = $(this).val();
  if (selected_unit == "ft") {
    $(".extra-field").show();
    if( !$("#height").val() ) {
    }
    else{
      var cm = $("#height").val();
      var ft_dec = cm / 30.48;
      var ft = Math.trunc(ft_dec);
      var inch = (ft_dec - ft) * 12;

      $("#height").val(ft);
      $("#in-input").val(Math.round(inch));
    }
  }
  else {
    $(".extra-field").hide();
    if( !$("#height").val() ) {
    }
    else{
      var ft = parseInt($("#height").val());
      var inch = $("#in-input").val();
      var ft_dec = ft + (inch / 12);
      var cm = ft_dec * 30.48;
      
      $("#height").val(Math.round(cm));

    }

  }
});
