$(document).on("click","#trigger-btn",function(){
  $("#file-select").trigger("click");
});

$(document).on("change","#file-select",function(){

  $("#show-filename").val($(this)[0].files[0].name);
  $("#SAR-calc").show();
});
$(document).on("click", "#computer-im",function(){
  location.href='recon';
});
$(document).on("click", "#rx-im",function(){
  location.href='rx';
});
$(document).on("click",".feedback-btn",function(){
  window.open('https://docs.google.com/forms/d/1267utGFl5VPDLE_6lQu153tF4vSTDTi4Kni9uam_QsM/edit', '_blank');
});
