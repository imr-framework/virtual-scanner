function selectFile(){

  $('#file-select').trigger('click');
  $('#file-select').change(function(){
    //var filePath = $('#file-select').val();
    var filePath = $('#file-select')[0].files[0].name;
    $("#file-input-display").val(filePath);
    $('#data-container').load(filePath);

});


}
