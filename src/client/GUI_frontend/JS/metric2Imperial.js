function metric2Imperial(){
  if($('#imp').is(':checked')){
        $("#weightlabel").html("lbs");
        $("#heightlabel").html("ft");
        $("#extraImpFields").show();
    }
  else if($('#met').is(':checked')){
    $("#weightlabel").html("kg");
    $("#heightlabel").html("cm");
    $("#extraImpFields").hide();
  }
}
