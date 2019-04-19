$(document).on("mouseover",".standard-tag",function(){

  $("#login-section").hide();
  $("#standard-about").show();
});
$(document).on("mouseout",".standard-tag",function(){

  $("#login-section").show();
  $("#standard-about").hide();
});

$(document).on("mouseover",".advanced-tag",function(){

  $("#login-section").hide();
  $("#advanced-about").show();
});
$(document).on("mouseout",".advanced-tag",function(){

  $("#login-section").show();
  $("#advanced-about").hide();
});
