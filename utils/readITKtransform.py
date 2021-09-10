import numpy

def readITKtransform( transform_file ):

    # read the transform
    transform = None
    with open( transform_file, 'r' ) as f:
      for line in f:

        # check for Parameters:
        if line.startswith( 'Parameters:' ):
          values = line.split( ': ' )[1].split( ' ' )

          # filter empty spaces and line breaks
          values = [float( e ) for e in values if ( e != '' and e != '\n' )]
          # create the upper left of the matrix
          transform_upper_left = numpy.reshape( values[0:9], ( 3, 3 ) )
          # grab the translation as well
          translation = numpy.reshape(values[9:], (3,1))

        # check for FixedParameters:
        if line.startswith( 'FixedParameters:' ):
          values = line.split( ': ' )[1].split( ' ' )

          # filter empty spaces and line breaks
          values = [float( e ) for e in values if ( e != '' and e != '\n' )]
          # setup the center
          center = numpy.reshape(values, (3,1))

    # compute the offset
    #offset = numpy.ones( 4 )
    offset = (center + translation) - transform_upper_left.dot(center)
    offset = numpy.vstack( ( offset, [1] ) )

    # add the [0, 0, 0] line
    transform = numpy.vstack( ( transform_upper_left, [0, 0, 0] ) )
    # and the [offset, 1] column
    transform = numpy.hstack( ( transform, numpy.reshape( offset, ( 4, 1 ) ) ) )

    return transform