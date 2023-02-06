from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, IntegerField, PasswordField, DecimalField, \
                    DecimalRangeField, SelectField, RadioField, IntegerRangeField
from wtforms.validators import DataRequired, EqualTo, Email, NumberRange, InputRequired


# Accounts and Authentication
class HalbachForm(FlaskForm):
    #
    ring_radii_field = StringField('Inner ring Radii (mm) ',default='148, 151, 154, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201')
    num_magnets_field = StringField('Numbers of inner ring magnets',default='50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69')
    num_rings_field = IntegerField('Number of rings',default=7)
    ring_separation_field = DecimalField('Ring separation (mm)',default=22)
    dsv_field = DecimalField('Diameter of Spherical Volume (mm)',default=200)

    # Display options
    xslice_field = IntegerField('x index')
    yslice_field = IntegerField('y index')
    zslice_field = IntegerField('z index')


    # username_field = StringField("Username",validators=[DataRequired()])
    # password_field = PasswordField('Password', validators=[DataRequired()])
    # password2_field = PasswordField('Re-enter Password',
    #                                 validators=[DataRequired(), EqualTo('password_field',message='Passwords must match')])
    # submit_field = SubmitField("Register!")

