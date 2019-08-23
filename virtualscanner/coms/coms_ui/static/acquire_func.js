/*
"""
This script contains several JavaScript functions related to the Acquire tab of the Virtual Scanner Standard mode.
----------

Author: Marina Manso;
Date: 04/12/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
*/
//This function calculates the voxel size depending on the selected matrix size
//and FOV. It also autocompletes N and FOV to only allow for square images.
$(document).ready(() => {

//This function calculates the voxel size depending on the selected matrix size
//and FOV. It also autocompletes N and FOV to only allow for square images.
$("#Nx").on("change",function(){
  var Nx = $('#Nx').val();
  $('#Ny').val(Nx);
  voxelSizeCalc();
})
$("#Ny").on("change",function(){
  var Ny = $('#Ny').val();
  $('#Nx').val(Ny);
  voxelSizeCalc();
})
$("#FOVx").on("change",function(){
  var FOVx = $('#FOVx').val();
  $('#FOVy').val(FOVx);
  voxelSizeCalc();
})
$("#FOVy").on("change",function(){
  var FOVy = $('#FOVy').val();
  $('#FOVx').val(FOVy);
  voxelSizeCalc();
})
$("#thickness").on("change",function(){
  voxelSizeCalc()
})
});

/*Voxel size calculation based on inputs for matrix size and FOV*/
function voxelSizeCalc() {

  var Nx = $('#Nx').val();
  var FOVx = $('#FOVx').val();
  var Ny = $('#Ny').val();
  var FOVy = $('#FOVy').val();
  var thck = $("#thickness").val();

  var val1 = FOVx/Nx;
  var val2 = FOVy/Ny;

  if ((Nx == 0)|| (FOVx == 0)){
    val1 = 0;
  }

  if ((Ny == 0),(FOVy == 0)){
    val2 = 0;
  }

  $("#vsx").html(val1.toFixed(2));
  $("#vsy").html(val2.toFixed(2));
  $("#thick").html(thck);
}

/*Function to open a new dialog to select the desired sequence everytime the + button is pressed*/
$(document).on("click", "#addseq-btn", function(){

  //$("#sl-gap").val(0);
  $(".dialog").dialog({
    open : function (ui){
      $( "#enclosingjumbo" ).addClass( "blur" );
      $( "#menu" ).menu();
    }
  });
});

/*Whenever an item from the sequence menu is selected it is added to the sequence history panel in the main document*/
$(document).on("click", ".menu-item", function(){

  $("#select-seq-sentence").hide();

  var seq_name = $(this).html();

  $("#seq-id").val(seq_name);
  $(".submit-form-btn").prop('disabled',false);
  /*If the selected sequence is SE show the option of IRSE*/
  if (seq_name == "SE") {
    $(".IRSE-checkbox").show();
    if ($(this).is(':checked')) {
      $(".TI-IRSE").show();
      $("#TI").prop("disabled",false);

      var help_link = "http://www.mriquestions.com/what-is-ir.html"

    }

    else if ($(this).is(':checked') == false) {
      $(".TI-IRSE").hide();
      $("#TI").prop("disabled",true);
      var help_link = "http://www.mriquestions.com/se-vs-multi-se-vs-fse.html"
    }
  }
  else {
    $(".IRSE-checkbox").hide();
    $(".TI-IRSE").hide();
    $("#TI").prop("disabled",true);

    if (seq_name == "GRE") {
      var help_link = "http://www.mriquestions.com/types-of-gre-sequences.html"
    }

  }

  $(".help-link").html("<a href="+help_link+">Q&As in MRI:"+seq_name+"</a>")

  var addlistitem_seq_name = "<li><a href='#'>" + seq_name + "</a></li>";



  $("#seq-history-list").append(addlistitem_seq_name);

  $("#dialog").dialog("close");


  //Default parameters

  $("#TR").val(200);
  $("#TE").val(10);
  $("#FA").val(90);
  $("#Nx").val(15);
  $("#Ny").val(15);




  $("#thickness").val(16);
  $("#ADC-bw").val(100);


  $("#slices").val(1);
  //$("#slices").prop("readonly", true);

  $("#FOVx").val(240);
  //$("#FOVx").prop("readonly", true);
  $("#FOVy").val(240);
  //$("#FOVy").prop("readonly", true);

  voxelSizeCalc();

  /*Gray-out gap field*/
  $("#sl-gap").css("background-color","#bfbfbf");
});

/*Add TI if IRSE is checked and hide it otherwise*/
$(document).on("change", "#IRSE-check", function(){
  if ($("#IRSE-check").is(':checked')) {
    $(".TI-IRSE").show();
    $("#TI").prop("disabled",false);
    $("#TI").val(20);
  }

  else if ($("#IRSE-check").is(':checked') == false) {
    $(".TI-IRSE").hide();
    $("#TI").prop("disabled",true);
  }

});

/*Blur and dialog opening/clositn effects*/
$(document).on("dialogclose","#dialog",function(){
  closeDialog();
});

function openDialog(){
  $( ".enclosingjumbo" ).addClass( "blur" );
  $( "#menu" ).menu();
}
function closeDialog(){
  $( "#enclosingjumbo" ).removeClass( "blur" );

}
$(document).on("click", ".submit-form-btn", function(){
  // TODO: if a seq is not selected abort submission
  $("#seqparameters")[0].submit();
});

function autoFillForm(dict){
  $("#TR").val(dict['TR']);
  $("#TE").val(dict['TE']);
  $("#FA").val(dict['FA']);
  if (dict['TI']){
    $("#TI").val(dict['TI']);
  }
  $("#sl-orient").val(dict['sl-orient']);
  var slorient = dict['sl-orient'];
  if (slorient == "axial"){
    $(".freq#x").prop("disabled",false);
    $("#freq").val("x");
    $(".phase#y").prop("disabled",false);
    $("#phase").val("y");
    $(".freq:not(#x)").prop("disabled",true);
    $(".phase:not(#y)").prop("disabled",true);
  }
  else if (slorient == "sagittal") {
    $(".freq#y").prop("disabled",false);
    $("#freq").val("y");
    $(".phase#z").prop("disabled",false);
    $("#phase").val("z");
    $(".freq:not(#y)").prop("disabled",true);
    $(".phase:not(#z)").prop("disabled",true);
  }
  else if (slorient == "coronal") {
    $(".freq#z").prop("disabled",false);
    $("#freq").val("z");
    $(".phase#x").prop("disabled",false);
    $("#phase").val("x");
    $(".freq:not(#z)").prop("disabled",true);
    $(".phase:not(#x)").prop("disabled",true);
  }
  $("#thickness").val(dict['thck']);
  $("#slices").val(dict['slicenum']);
  $("#sl-gap").prop("disabled",true);
  $("#sl-gap").css("background-color","#bfbfbf");

  $("#ADC-bw").val(dict['bw']);
  $("#Nx").val(dict['Nx']);
  $("#Ny").val(dict['Ny']);
  $("#FOVx").val(dict['FOVx']);
  $("#FOVy").val(dict['FOVy']);
  $("#select-seq-sentence").hide();
  $("#seq-history-list").append(dict['selectedSeq']);
}

$(document).on("change", "#sl-orient",function(){
  var val = $(this).val();

  if (val == "axial"){
    $(".freq#x").prop("disabled",false);
    $("#freq").val("x");
    $(".phase#y").prop("disabled",false);
    $("#phase").val("y");
    $(".freq:not(#x)").prop("disabled",true);
    $(".phase:not(#y)").prop("disabled",true);
  }
  else if (val == "sagittal") {
    $(".freq#y").prop("disabled",false);
    $("#freq").val("y");
    $(".phase#z").prop("disabled",false);
    $("#phase").val("z");
    $(".freq:not(#y)").prop("disabled",true);
    $(".phase:not(#z)").prop("disabled",true);
  }
  else if (val == "coronal") {
    $(".freq#z").prop("disabled",false);
    $("#freq").val("z");
    $(".phase#x").prop("disabled",false);
    $("#phase").val("x");
    $(".freq:not(#z)").prop("disabled",true);
    $(".phase:not(#x)").prop("disabled",true);
  }
});
$(document).on("click",".feedback-btn",function(){
  window.open('https://docs.google.com/forms/u/2/d/15kb_3yJE5vYiTZo-N1pTA576atc3tjth58lud1S4MVA/edit?usp=sharing_eip&ts=5cf6c4a7', '_blank');
});
$(document).on("change","#TR",function(){

  $("#TE").attr("max",$(this).val());
});
$(function () {
  $(".param-with-range").keydown(function () {
    // Save old value.
    if (!$(this).val() || (parseInt($(this).val()) <= $(this).attr('max') && parseInt($(this).val()) >= $(this).attr('min')))

    $(this).data("old", $(this).val());
  });
  $(".param-with-range").keyup(function () {
    // Check correct, else revert back to old value.
    if (!$(this).val() || (parseInt($(this).val()) <= $(this).attr('max') && parseInt($(this).val()) >= $(this).attr('min')))
      ;
    else
      $(this).val($(this).data("old"));
  });
});

// TODO: change this to
//image right-half: slice advance and viceversa
/*Multi-slice slider
$(document).ready(() => {

  $('#axial-slider').on("change",function(){
    var sl_number = $('#axial-slider').val();
    $("#axial-container").html("<img class='square-img' src='../static/IMG_0" + sl_number + ".jpg'/>");

  });
});
*/
/*$( document ).on("change","#dialog",function(ui){
   if ($(this).dialog() == "open") {

   }
    $( "#dialog" ).dialog({
        "open": function() {
            $( ".enclosingjumbo" ).addClass( "blur" );
            $( "#menu" ).menu();
        },
        "close": function() {
            $( ".text-box" ).removeClass( "blur" );
        }
    });

});*/
