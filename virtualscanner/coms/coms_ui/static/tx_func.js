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
