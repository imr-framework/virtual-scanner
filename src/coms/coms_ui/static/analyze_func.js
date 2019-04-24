/*Menu opens when the load button is pressed*/
$(document).on("click","#load-btn",function(){
  $("#file-select-dialog").dialog({
    open : function (ui){
      $( "#enclosingjumbo" ).addClass( "blur" );
      $( "#menu" ).menu();
    }
  });
});

/*When an item of the menu is selected:
1.get the folder Name
2.get the list of files inside the folder
*/
$(document).on("click", ".menu-item", function(){

  var folder_name = $(this).html();

  if (folder_name == "Analyze Sample Data"){
    var folder_path = "../static/Analyze_Sample/"

  }
  $('.file-list').html("<img src='"+folder_path+"pic_1.")


});

/*Enable/Disable TI input field based on sequence choice*/
$(document).on("change", "#seq-choice", function(){
  $("#TI-input").val("");
  if ($(this).val() == "SE") {
    $("#TI-input").css("background-color","#bfbfbf");
    $("#TI-input").prop("readonly",true);
  }
  else{
    $("#TI-input").css("background-color","white");
    $("#TI-input").prop("readonly",false);
  }
});



/*function selectFile(){

  $('#file-select').trigger('click');
  $('#file-select').change(function(){
    //var filePath = $('#file-select').val();
    var filePath = $('#file-select')[0].files[0].name;
    $("#file-input-display").val(filePath);
    $('#data-container').load(filePath);

});*/
