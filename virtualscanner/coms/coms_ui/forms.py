from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, IntegerField, PasswordField, DecimalField, \
                    DecimalRangeField, SelectField, RadioField, IntegerRangeField
from wtforms.validators import DataRequired, EqualTo, Email, NumberRange, InputRequired


# Accounts and Authentication
class HalbachForm(FlaskForm):
    # Environment
    temp_sensitivity_field = DecimalField('Temp. coefficient of remanence (ppm/K)', default=0)
    temperature_field = DecimalRangeField('Temperature (C)', validators= [DataRequired(), NumberRange(min=-50, max=50)],default=25)

    #
    ring_radii_field = StringField('Inner ring Radii (mm) ',default='148, 151, 154')#, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201')
    num_magnets_field = StringField('Numbers of inner ring magnets',default='50, 51, 52') #, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69')
    num_rings_field = IntegerField('Number of rings',default=3)
    ring_separation_field = DecimalField('Ring separation (mm)',default=22)
    dsv_field = DecimalField('Diameter of Spherical Volume (mm)',default=50)
    resolution_field = DecimalField('Resolution (mm)', default=5)
    max_gen_field = IntegerField("Max number of generations",default=10)

    # Display options
    dsv_display_field = DecimalField('Display DSV (mm)', default=50)
    res_display_field = DecimalField('Display resolution (mm)', default=5)
    xslice_field = IntegerField('x index',default=5)
    yslice_field = IntegerField('y index',default=5)
    zslice_field = IntegerField('z index',default=5)

    # 3D orientation
    options_3d = RadioField('3D slicing', choices=['X', 'Y', 'Z'],validators=[],default='Z')

    # Load options
    load_options_field = SelectField('Loading option', choices=[('default','Default'),
                                                                ('prevsim','halbach.mat')], default='default')



    # username_field = StringField("Username",validators=[DataRequired()])
    # password_field = PasswordField('Password', validators=[DataRequired()])
    # password2_field = PasswordField('Re-enter Password',
    #                                 validators=[DataRequired(), EqualTo('password_field',message='Passwords must match')])
    # submit_field = SubmitField("Register!")


class SequenceForm(FlaskForm):
    min_time_field = DecimalField('Start time (s)', default=0)
    max_time_field = DecimalField('End time (s)', default=2)
