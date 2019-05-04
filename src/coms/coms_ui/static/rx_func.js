$(document).on("change", "#orig-im", function(){
  var im_orient = $(this).val();
  changeOrigIm(im_orient);
});
function changeOrigIm(im_orient){
  $("#"+im_orient).show();
  switch (im_orient) {
    case "axial":
      $("#coronal").hide();
      $("#sagittal").hide();
      break;
    case "coronal":
      $("#axial").hide();
      $("#sagittal").hide();
      break;
    case "sagittal":
      $("#coronal").hide();
      $("#axial").hide();
      break;
    default:

  }
}
function autoFillForm(dict){
  
  var im_orient = dict['image-or'];
  $("#orig-im").val(im_orient);
  changeOrigIm(im_orient);
  $("#dsf").val(dict['DSF']);
  $("#deltaf").val(dict['deltaf']);
}