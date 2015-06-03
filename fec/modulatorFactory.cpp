
#include "modulatorFactory.h"


SymbolTable::SymbolTable( int k, CODE_RATE rate, FRAME_TYPE framesize )
{
	this->k = k;
	this->rate = rate;
	this->framesize = framesize;

	int m = 1<<k;
	symbols.set_size( m );

	bits10symbols.set_size( m );
	bits2symbols.set_size( m );
	for (int i=0; i<m; i++)
	{
		bits10symbols[i] = i;
		bits2symbols[i] = dec2bin(k, i);
	}

	switch(k)
	{
	case 1:
		r1 = 1;

		symbols[0].real( r1 );
		symbols[0].imag( 0.0f );
		symbols[1].real( -r1 );
		symbols[1].imag( 0.0f );
		break;

	case 2:
		r1 = 1;

		symbols[0].real( r1 * cos(pi / 4.0));
		symbols[0].imag( r1 * sin(pi / 4.0));
		symbols[1].real( r1 * cos(-pi / 4.0));
		symbols[1].imag( r1 * sin(-pi / 4.0));
		symbols[2].real( r1 * cos(3 * pi / 4.0));
		symbols[2].imag( r1 * sin(3 * pi / 4.0));
		symbols[3].real( r1 * cos(-3 * pi / 4.0));
		symbols[3].imag( r1 * sin(-3 * pi / 4.0));

		break;

	case 3:
		r1 = 1;

		symbols[0].real( r1 * cos(pi / 4.0));
		symbols[0].imag( r1 * sin(pi / 4.0));
		symbols[1].real( r1 );
		symbols[1].imag( 0.0f );
		symbols[2].real( -r1 );
		symbols[2].imag( 0.0f );
		symbols[3].real( r1 * cos(-3 * pi / 4.0));
		symbols[3].imag( r1 * sin(-3 * pi / 4.0));
		symbols[4].real( 0.0f );
		symbols[4].imag( r1 );
		symbols[5].real( r1 * cos(-pi / 4.0));
		symbols[5].imag( r1 * sin(-pi / 4.0));
		symbols[6].real( r1 * cos(3 * pi / 4.0));
		symbols[6].imag( r1 * sin(3 * pi / 4.0));
		symbols[7].real( 0.0f );
		symbols[7].imag( -r1 );

		break;

	case 4:
		r2 = 1.0f;
		if (framesize == FECFRAME_NORMAL)
		{
			switch(rate)
			{
			case C2_3:
				r1 = r2 / 3.15;
				break;
			case C3_4:
				r1 = r2 / 2.85;
				break;
			case C4_5:
				r1 = r2 / 2.75;
				break;
			case C5_6:
				r1 = r2 / 2.70;
				break;
			case C8_9:
				r1 = r2 / 2.60;
				break;
			case C9_10:
				r1 = r2 / 2.57;
				break;

			default:
				r1 = 0;
				break;
			}
		}
		else
		{
			switch(rate)
			{
			case C2_3:
				r1 = r2 / 3.15;
				break;
			case C3_4:
				r1 = r2 / 2.85;
				break;
			case C4_5:
				r1 = r2 / 2.75;
				break;
			case C5_6:
				r1 = r2 / 2.70;
				break;
			case C8_9:
				r1 = r2 / 2.60;
				break;

			default:
				r1 = 0;
				break;
			}
		}


		symbols[0].real( r2 * cos(pi / 4.0));
		symbols[0].imag( r2 * sin(pi / 4.0));
		symbols[1].real( r2 * cos(-pi / 4.0));
		symbols[1].imag( r2 * sin(-pi / 4.0));
		symbols[2].real( r2 * cos(3 * pi / 4.0));
		symbols[2].imag( r2 * sin(3 * pi / 4.0));
		symbols[3].real( r2 * cos(-3 * pi / 4.0));
		symbols[3].imag( r2 * sin(-3 * pi / 4.0));
		symbols[4].real( r2 * cos(pi / 12.0));
		symbols[4].imag( r2 * sin(pi / 12.0));
		symbols[5].real( r2 * cos(-pi / 12.0));
		symbols[5].imag( r2 * sin(-pi / 12.0));
		symbols[6].real( r2 * cos(11 * pi / 12.0));
		symbols[6].imag( r2 * sin(11 * pi / 12.0));
		symbols[7].real( r2 * cos(-11 * pi / 12.0));
		symbols[7].imag( r2 * sin(-11 * pi / 12.0));
		symbols[8].real( r2 * cos(5 * pi / 12.0));
		symbols[8].imag( r2 * sin(5 * pi / 12.0));
		symbols[9].real( r2 * cos(-5 * pi / 12.0));
		symbols[9].imag( r2 * sin(-5 * pi / 12.0));
		symbols[10].real( r2 * cos(7 * pi / 12.0));
		symbols[10].imag( r2 * sin(7 * pi / 12.0));
		symbols[11].real( r2 * cos(-7 * pi / 12.0));
		symbols[11].imag( r2 * sin(-7 * pi / 12.0));
		symbols[12].real( r1 * cos(pi / 4.0));
		symbols[12].imag( r1 * sin(pi / 4.0));
		symbols[13].real( r1 * cos(-pi / 4.0));
		symbols[13].imag( r1 * sin(-pi / 4.0));
		symbols[14].real( r1 * cos(3 * pi / 4.0));
		symbols[14].imag( r1 * sin(3 * pi / 4.0));
		symbols[15].real( r1 * cos(-3 * pi / 4.0));
		symbols[15].imag( r1 * sin(-3 * pi / 4.0));

		break;

	case 5:
		r3 = 1.0f;

		switch(rate)
		{
		case C3_4:
			r1 = r3 / 5.27;
			r2 = r1 * 2.84;
			break;
		case C4_5:
			r1 = r3 / 4.87;
			r2 = r1 * 2.72;
			break;
		case C5_6:
			r1 = r3 / 4.64;
			r2 = r1 * 2.64;
			break;
		case C8_9:
			r1 = r3 / 4.33;
			r2 = r1 * 2.54;
			break;
		case C9_10:
			r1 = r3 / 4.30;
			r2 = r1 * 2.53;
			break;
		default:
			r1 = 0;
			r2 = 0;
			break;
		}

		symbols[0].real( r2 * cos(pi / 4.0) );
		symbols[0].imag( r2 * sin(pi / 4.0));
		symbols[1].real( r2 * cos(5 * pi / 12.0));
		symbols[1].imag( r2 * sin(5 * pi / 12.0));
		symbols[2].real( r2 * cos(-pi / 4.0));
		symbols[2].imag( r2 * sin(-pi / 4.0));
		symbols[3].real( r2 * cos(-5 * pi / 12.0));
		symbols[3].imag( r2 * sin(-5 * pi / 12.0));
		symbols[4].real( r2 * cos(3 * pi / 4.0));
		symbols[4].imag( r2 * sin(3 * pi / 4.0));
		symbols[5].real( r2 * cos(7 * pi / 12.0));
		symbols[5].imag( r2 * sin(7 * pi / 12.0));
		symbols[6].real( r2 * cos(-3 * pi / 4.0));
		symbols[6].imag( r2 * sin(-3 * pi / 4.0));
		symbols[7].real( r2 * cos(-7 * pi / 12.0));
		symbols[7].imag( r2 * sin(-7 * pi / 12.0));
		symbols[8].real( r3 * cos(pi / 8.0));
		symbols[8].imag( r3 * sin(pi / 8.0));
		symbols[9].real( r3 * cos(3 * pi / 8.0));
		symbols[9].imag( r3 * sin(3 * pi / 8.0));
		symbols[10].real( r3 * cos(-pi / 4.0));
		symbols[10].imag( r3 * sin(-pi / 4.0));
		symbols[11].real( r3 * cos(-pi / 2.0));
		symbols[11].imag( r3 * sin(-pi / 2.0));
		symbols[12].real( r3 * cos(3 * pi / 4.0));
		symbols[12].imag( r3 * sin(3 * pi / 4.0));
		symbols[13].real( r3 * cos(pi / 2.0));
		symbols[13].imag( r3 * sin(pi / 2.0));
		symbols[14].real( r3 * cos(-7 * pi / 8.0));
		symbols[14].imag( r3 * sin(-7 * pi / 8.0));
		symbols[15].real( r3 * cos(-5 * pi / 8.0));
		symbols[15].imag( r3 * sin(-5 * pi / 8.0));
		symbols[16].real( r2 * cos(pi / 12.0));
		symbols[16].imag( r2 * sin(pi / 12.0));
		symbols[17].real( r1 * cos(pi / 4.0));
		symbols[17].imag( r1 * sin(pi / 4.0));
		symbols[18].real( r2 * cos(-pi / 12.0));
		symbols[18].imag( r2 * sin(-pi / 12.0));
		symbols[19].real( r1 * cos(-pi / 4.0));
		symbols[19].imag( r1 * sin(-pi / 4.0));
		symbols[20].real( r2 * cos(11 * pi / 12.0));
		symbols[20].imag( r2 * sin(11 * pi / 12.0));
		symbols[21].real( r1 * cos(3 * pi / 4.0));
		symbols[21].imag( r1 * sin(3 * pi / 4.0));
		symbols[22].real( r2 * cos(-11 * pi / 12.0));
		symbols[22].imag( r2 * sin(-11 * pi / 12.0));
		symbols[23].real( r1 * cos(-3 * pi / 4.0));
		symbols[23].imag( r1 * sin(-3 * pi / 4.0));
		symbols[24].real( r3 * cos(0.0));
		symbols[24].imag( r3 * sin(0.0));
		symbols[25].real( r3 * cos(pi / 4.0));
		symbols[25].imag( r3 * sin(pi / 4.0));
		symbols[26].real( r3 * cos(-pi / 8.0));
		symbols[26].imag( r3 * sin(-pi / 8.0));
		symbols[27].real( r3 * cos(-3 * pi / 8.0));
		symbols[27].imag( r3 * sin(-3 * pi / 8.0));
		symbols[28].real( r3 * cos(7 * pi / 8.0));
		symbols[28].imag( r3 * sin(7 * pi / 8.0));
		symbols[29].real( r3 * cos(5 * pi / 8.0));
		symbols[29].imag( r3 * sin(5 * pi / 8.0));
		symbols[30].real( r3 * cos(pi));
		symbols[30].imag( r3 * sin(pi));
		symbols[31].real( r3 * cos(-3 * pi / 4.0));
		symbols[31].imag( r3 * sin(-3 * pi / 4.0));

		break;

	default:
		break;
	}
}

ModulatorFactory::ModulatorFactory()
{
	for (int i=MOD_DEFAULT+1; i<MOD_COUNT; i++)
	{
		MOD_TYPE modType = (MOD_TYPE)i;
		SymbolTable symbolItem(modType);
		Modulator_2D* pModulator = new Modulator_2D( symbolItem.getSymbols(), symbolItem.getBits10Symbols() );

		m_modPool.insert( ModPoolPair(modType, pModulator) );
	}
}

ModulatorFactory::~ModulatorFactory()
{
	for (ModPoolItr itr=m_modPool.begin(); itr!=m_modPool.end(); itr++)
		delete itr->second;

	m_modPool.clear();
}

Modulator_2D* ModulatorFactory::findModulator( MOD_TYPE modType )
{
	ModPoolItr itr=m_modPool.find( modType );
	if ( itr != m_modPool.end() )
		return itr->second;
	else
		return NULL;
}
