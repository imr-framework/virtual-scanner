$(document).on("click","#trigger-btn",function(){
  $("#file-select").trigger("click");
});
$(document).on("change","#file-select",function(){

  $("#show-filename").val($(this)[0].files[0].name);
  $("#submit-btn").show();
});
$(document).on("click", "#tx-im",function(){
  location.href='tx';
});
$(document).on("click", "#rx-im",function(){
  location.href='rx';
});

function autoFillForm(dict){
  var dl_type = dict['DL-type'];
  $("#"+dl_type).val(dl_type);
  $("#show-filename").val(dict['file-input-name']);

}
