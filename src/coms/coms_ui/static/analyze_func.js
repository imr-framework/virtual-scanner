/*Menu opens when the load button is pressed*/
$(document).on("click","#load-btn",function(){
  $(".dialog").dialog({
    open : function (ui){
      $( "#enclosingjumbo" ).addClass( "blur" );
      $( ".menu" ).menu();
      $('.menu').css('width','auto')
    }
  });
});
/*Dialog closing and bluring*/
$(document).on("dialogclose",".dialog",function(){
  closeDialog();
});

function openDialog(){
  $( ".enclosingjumbo" ).addClass( "blur" );
  $( "#menu" ).menu();
}
function closeDialog(){
  $( "#enclosingjumbo" ).removeClass( "blur" );

}

/*When an item of the menu is selected:*/
$(document).on("click", ".menu-item", function(){

  //Selected option to html template
  $("#selected-option").val($(this).html());
  $(".dialog").dialog("close");
  $("#original-data-form")[0].submit();
  /*setTimeout(function(){
    $("#original-data-form")[0].submit();
  }, 500);*/


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

function autoFillForm(dict){
  $("#file-input-display").val('Virtual Scanner/Analyze/Sample Data/'+dict['original-data-opt']);
  if (dict['original-data-opt'] == 'T1'){

    $("#seq-choice").val('IRSE');
    $("#TE").val(12);
    $("#TI-input").css('background-color','white');
    $("#TI-input").val('21, 100, 200, 400, 800, 1600, 3200');
    $("#model-eq").val('IRSE-eq');
    $("#map-type").val('t1');
    $("#FOV").val(170);

  }
  else if (dict['original-data-opt'] == 'T2') {
    $("#seq-choice").val('SE');
    $("#TE").val('12, 22, 42, 62, 102, 152, 202');
    $("#model-eq").val('SE-eq');
    $("#map-type").val('t2');
    $("#FOV").val(210);
  }
  $('#seq-choice option:not(:selected)').attr('disabled', true);
  $('#model-eq option:not(:selected)').attr('disabled', true);
  $('#map-type option:not(:selected)').attr('disabled', true);
  $("#TR").val(10000);
  $("#map-size").val(128);


}
function autoFillForm2(dict1,dict2){
  //$("#file-input-display").val('Virtual Scanner/Analyze/Sample Data/'+dict1['original-data-opt']);
  //$("#seq-choice").val(dict2['seq-choice']);
  $("#TR").val(dict2['TR']);
  $("#TE").val(dict2['TE']);
  $("#TI-input").val(dict2['TI']);
  //$("#model-eq").val(dict2['model-eq']);
  //$("#map-size").val(128);

  //if (dict1['original-data-opt'] == 'T1'){
    //$("#TI-input").css('background-color','white');
    //$("#map-type").val('t1');
    //$("#FOV").val(170);

  //}
  //else if (dict1['original-data-opt'] == 'T2') {
    //$("#map-type").val('t2');
    //$("#FOV").val(210);
  //}
  //$('#seq-choice option:not(:selected)').attr('disabled', true);
  //$('#model-eq option:not(:selected)').attr('disabled', true);
  //$('#map-type option:not(:selected)').attr('disabled', true);
  $('#roi-analysis-button').prop('disabled',false);
  $('#roi-analysis-button').css('background-color','#5f9bef');
}






/*function selectFile(){

  $('#file-select').trigger('click');
  $('#file-select').change(function(){
    //var filePath = $('#file-select').val();
    var filePath = $('#file-select')[0].files[0].name;
    $("#file-input-display").val(filePath);
    $('#data-container').load(filePath);

});*/
