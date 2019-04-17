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

  var val1 = Nx/FOVx;
  var val2 = Ny/FOVy;

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
  $("#dialog").dialog({
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

  /*If the selected sequence is SE show the option of IRSE*/
  if (seq_name == "SE") {
    $(".IRSE-checkbox").show();
  }
  else {
    $(".IRSE-checkbox").hide();
  }

  var addlistitem_seq_name = "<li><a href='#'>" + seq_name + "</a></li>";



  $("#seq-history-list").append(addlistitem_seq_name);

  $("#dialog").dialog("close");
  closeDialog();
  /*$("#dialog").dialog({
    "close" : function (ui){
      $( "#enclosingjumbo" ).removeClass( "blur" );

    }
  });*/
  //Default parameters
  $("#slices").val(1);
  $("#slices").prop("readonly", true);
  $("#Nx").val(16);
  $("#Nx").prop("readonly", true);
  $("#Ny").val(16);
  $("#Ny").prop("readonly", true);
  $("#FOVx").val(240);
  $("#FOVx").prop("readonly", true);
  $("#FOVy").val(240);
  $("#FOVy").prop("readonly", true);
  voxelSizeCalc();

});

/*Add TI if IRSE is checked and hide it otherwise*/
$(document).on("change", "#IRSE-check", function(){
  if ($(this).is(':checked')) {
    $(".TI-IRSE").show();
  }

  else if ($(this).is(':checked') == false) {
    $(".TI-IRSE").hide();
  }

});

/*Blur and dialog opening/clositn effects*/
function OpenDialog(){
  $( ".enclosingjumbo" ).addClass( "blur" );
  $( "#menu" ).menu();
}
function closeDialog(){
  $( "#enclosingjumbo" ).removeClass( "blur" );

}

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
