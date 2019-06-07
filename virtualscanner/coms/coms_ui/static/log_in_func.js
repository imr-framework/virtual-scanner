$(document).on("click",".mode-option",function(){

  if ($(this).val() == "Standard") {
    $("#standard-about").show();
    $("#advanced-about").hide();
  }
  else {
    $("#standard-about").hide();
    $("#advanced-about").show(); 
  }

});
